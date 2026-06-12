"""Reconstruct (dequantize) a quantized model back to fp16.

Usage:
    python reconstruct.py --quantized_dir ./quantized_llama2-7b --output_dir ./reconstructed
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq_sinkhorn.algorithms.gptq import dequantize_layer
from gptq_sinkhorn.config import QuantizationConfig, QuantizedLayerData
from gptq_sinkhorn.models import detect_and_get_handler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quantized_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    quantized_dir = Path(args.quantized_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(quantized_dir / "quant_config.json") as f:
        cfg = QuantizationConfig.from_dict(json.load(f))

    print(f"Loading base model {cfg.model_name} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(quantized_dir / "tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()

    handler = detect_and_get_handler(model)
    layers = handler.get_layers(model)
    layer_count = len(layers)

    # Restore global (non-quantized) params
    global_params = torch.load(quantized_dir / "global_params.pt", weights_only=True)
    for path, emb in handler.get_embeddings(model).items():
        for pname, param in emb.named_parameters():
            key = f"{path}.{pname}"
            if key in global_params:
                param.data.copy_(global_params[key])
    final_norm = handler.get_final_norm(model)
    for pname, param in final_norm.named_parameters():
        key = f"final_norm.{pname}"
        if key in global_params:
            param.data.copy_(global_params[key])
    if hasattr(model, "lm_head"):
        for pname, param in model.lm_head.named_parameters():
            key = f"lm_head.{pname}"
            if key in global_params:
                param.data.copy_(global_params[key])

    for layer_idx in range(layer_count):
        print(f"Reconstructing layer {layer_idx} / {layer_count - 1} ...", file=sys.stderr)
        layer = layers[layer_idx]
        layer_dir = quantized_dir / f"layer_{layer_idx}"

        linear_layers = handler.get_linear_layers(layer)
        for name, linear in linear_layers.items():
            pt_path = layer_dir / f"{name.replace('.', '_')}.pt"
            if not pt_path.exists():
                print(f"  WARNING: {pt_path} not found, skipping.", file=sys.stderr)
                continue
            quant_data = QuantizedLayerData.load(pt_path)
            W_fp16 = dequantize_layer(quant_data)
            linear.weight.data.copy_(W_fp16)
            if quant_data.bias is not None and linear.bias is not None:
                linear.bias.data.copy_(quant_data.bias.float())

        # Restore non-quantized params (norms, etc.)
        nq_path = layer_dir / "non_quantized.pt"
        if nq_path.exists():
            non_quant = torch.load(nq_path, weights_only=True)
            state = layer.state_dict()
            for pname, tensor in non_quant.items():
                if pname in state:
                    state[pname].copy_(tensor)

    print(f"Saving reconstructed model to {output_dir} ...", file=sys.stderr)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
