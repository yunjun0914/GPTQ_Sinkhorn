"""Perplexity evaluation on wikitext2 / c4.

Usage:
    # From reconstructed fp16 model:
    python eval_ppl.py --model_dir ./reconstructed_llama2-7b --dataset wikitext2 c4

    # From original HuggingFace model (baseline):
    python eval_ppl.py --model_name meta-llama/Llama-2-7b-hf --dataset wikitext2
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", help="Path to saved HuggingFace model directory")
    group.add_argument("--model_name", help="HuggingFace model name (for baseline)")
    p.add_argument("--dataset", nargs="+", default=["wikitext2"],
                   choices=["wikitext2", "c4"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def eval_ppl(model, tokenizer, dataset_name: str, seqlen: int, device) -> float:
    if dataset_name == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    elif dataset_name == "c4":
        data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text = " ".join(row["text"] for row in list(data)[:1100])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    n_tokens = enc.shape[0]
    n_windows = n_tokens // seqlen

    model.eval()
    model.to(device)

    nlls = []
    for i in range(n_windows):
        window = enc[i * seqlen : (i + 1) * seqlen].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(window, labels=window)
        nlls.append(out.loss.item())

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def main():
    args = parse_args()
    model_path = args.model_dir or args.model_name

    print(f"Loading model from {model_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for ds in args.dataset:
        print(f"Evaluating {ds} ...", file=sys.stderr)
        ppl = eval_ppl(model, tokenizer, ds, args.seqlen, device)
        print(f"{ds}: PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
