"""Main quantization script.

Usage:
    python quantize.py meta-llama/Llama-2-7b-hf --output_dir ./quantized_llama2-7b
    python quantize.py meta-llama/Llama-2-7b-hf --output_dir ./out --bits 4 --sinkhorn --hadamard_rotation
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq_sinkhorn.algorithms.gptq import gptq_quantize
from gptq_sinkhorn.config import QuantizationConfig
from gptq_sinkhorn.models import detect_and_get_handler
from gptq_sinkhorn.utils.calibration import get_calibration_data
from gptq_sinkhorn.utils.hessian import get_hessians, get_layer_output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQ-Sinkhorn quantization")
    p.add_argument("model_name", help="HuggingFace model name or local path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group_size", type=int, default=-1,
                   help="Columns per group for per-row-group scale. -1 = per-column.")
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calib_dataset", default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--sinkhorn", action="store_true", default=True,
                   help="Apply Sinkhorn row/col normalization (default: on)")
    p.add_argument("--no_sinkhorn", dest="sinkhorn", action="store_false")
    p.add_argument("--hadamard_rotation", action="store_true",
                   help="Apply randomized Hadamard rotation for incoherence")
    p.add_argument("--rotation_first", action="store_true",
                   help="Apply Hadamard rotation BEFORE Sinkhorn (default: Sinkhorn first)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = QuantizationConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        seed=args.seed,
        calib_dataset=args.calib_dataset,
        sinkhorn=args.sinkhorn,
        hadamard_rotation=args.hadamard_rotation,
        rotation_first=args.rotation_first,
        device=args.device,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {cfg.model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()

    handler = detect_and_get_handler(model)

    print(f"Loading calibration data ({cfg.calib_dataset}, {cfg.n_samples} samples) ...",
          file=sys.stderr)
    calib = get_calibration_data(
        cfg.calib_dataset, cfg.n_samples, cfg.seed, cfg.seqlen, tokenizer
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Move non-layer params to CPU, layers stay CPU until processed
    layers = handler.get_layers(model)
    layer_count = len(layers)

    print("Computing first-layer inputs ...", file=sys.stderr)
    # Move model (sans layers) to device for embedding computation
    embeddings = handler.get_embeddings(model)
    for emb in embeddings.values():
        emb.to(device)
    handler.get_final_norm(model).to(device)

    layer_input, cache = handler.get_first_layer_input(model, calib, batchsize=8)
    forward_kwargs = handler.get_layer_forward_kwargs(cache)

    # Move embeddings back to CPU to free VRAM
    for emb in embeddings.values():
        emb.cpu()

    for layer_idx in range(layer_count):
        print(f"\n=== Layer {layer_idx} / {layer_count - 1} ===", file=sys.stderr)
        layer = layers[layer_idx]
        layer.to(device)

        # Compute Hessians for all linear sublayers
        hessians = get_hessians(layer, layer_input, batchsize=4, forward_kwargs=forward_kwargs)

        linear_layers = handler.get_linear_layers(layer)

        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        for name, linear in linear_layers.items():
            print(f"  Quantizing {name} ...", file=sys.stderr)
            H = hessians[name]
            W = linear.weight.data.float()

            quant_data = gptq_quantize(
                H_orig=H,
                W_orig=W,
                bits=cfg.bits,
                percdamp=cfg.percdamp,
                blocksize=cfg.blocksize,
                group_size=cfg.group_size,
                sinkhorn=cfg.sinkhorn,
                hadamard_rotation=cfg.hadamard_rotation,
                rotation_first=cfg.rotation_first,
                layer_name=f"layer_{layer_idx}.{name}",
            )
            if linear.bias is not None:
                quant_data.bias = linear.bias.data.half().cpu()

            quant_data.save(layer_dir / f"{name.replace('.', '_')}.pt")

        # Save non-quantized layer params (norms, etc.)
        non_quant = {}
        linear_param_names = set()
        for name, linear in linear_layers.items():
            for pname, _ in linear.named_parameters():
                linear_param_names.add(f"{name}.{pname}")

        for pname, param in layer.named_parameters():
            if pname not in linear_param_names:
                non_quant[pname] = param.data.cpu()
        torch.save(non_quant, layer_dir / "non_quantized.pt")

        # Compute output for next layer
        layer_input = get_layer_output(layer, layer_input, batchsize=4, forward_kwargs=forward_kwargs)

        layer.cpu()
        torch.cuda.empty_cache()

    # Save non-layer params (embeddings, final norm, lm_head)
    print("\nSaving global params ...", file=sys.stderr)
    global_params = {}
    for path, emb in handler.get_embeddings(model).items():
        for pname, param in emb.named_parameters():
            global_params[f"{path}.{pname}"] = param.data.cpu()
    final_norm = handler.get_final_norm(model)
    for pname, param in final_norm.named_parameters():
        global_params[f"final_norm.{pname}"] = param.data.cpu()
    if hasattr(model, "lm_head"):
        for pname, param in model.lm_head.named_parameters():
            global_params[f"lm_head.{pname}"] = param.data.cpu()
    torch.save(global_params, output_dir / "global_params.pt")

    # Save config and tokenizer
    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    print(f"\nDone. Saved to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
