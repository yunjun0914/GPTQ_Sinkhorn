# CLAUDE.md

## Project Overview

GPTQ-Sinkhorn: LLM post-training quantization using GPTQ error propagation + symmetric RTN,
with optional Sinkhorn row/col normalization and randomized Hadamard rotation for incoherence.

## Commands

```bash
# Install
pip install -e .

# Quantize a model
python quantize.py meta-llama/Llama-2-7b-hf \
    --output_dir ./quantized_llama2-7b \
    --bits 4 \
    --sinkhorn \
    --hadamard_rotation

# Reconstruct (dequantize back to fp16)
python reconstruct.py \
    --quantized_dir ./quantized_llama2-7b \
    --output_dir ./reconstructed_llama2-7b

# Evaluate PPL
python eval_ppl.py --model_dir ./reconstructed_llama2-7b --dataset wikitext2 c4
```

## Architecture

### Pipeline (per linear layer)

```
W, H = 2·X^T·X / N
  → comp_gh(W)            [if --sinkhorn]    Sinkhorn → g, h  (row/col scales)
  → W /= g·h,  H *= h²
  → Hadamard(W, H)        [if --hadamard_rotation]  incoherence
  → GPTQ + sym RTN                           column-wise quantize + error propagation
```

### Key files

- `gptq_sinkhorn/algorithms/gptq.py`  — `comp_gh`, `gptq_quantize`, `dequantize_layer`
- `gptq_sinkhorn/config.py`           — `QuantizationConfig`, `QuantizedLayerData`
- `gptq_sinkhorn/utils/hadamard.py`   — Hadamard transform utilities
- `gptq_sinkhorn/utils/hessian.py`    — Hessian accumulation via hooks
- `gptq_sinkhorn/models/`             — LlamaHandler, OPTHandler

### QuantizedLayerData format

| Field      | Type  | Shape                         | Description                         |
|------------|-------|-------------------------------|-------------------------------------|
| `Q`        | int8  | (out, in)                     | Symmetric quantized indices         |
| `scales`   | fp16  | (out,) or (out, n_groups)     | Per-row or per-row-group scale      |
| `g`        | fp16  | (out,)                        | Sinkhorn row scale                  |
| `h`        | fp16  | (in,)                         | Sinkhorn col scale                  |
| `had_d`    | int8  | (in,)                         | Hadamard ±1 sign vector             |
| `had_K_col`| int   | scalar                        | Number of Hadamard blocks           |

### Reconstruction

```
W_q = Q * scales
    → undo Hadamard  (apply_inverse_had_to_W_single)
    → W = W_q * g[:, None] * h[None, :]
```

## Output Structure

```
quantized_model/
├── quant_config.json
├── global_params.pt          # embeddings, final norm, lm_head
├── tokenizer/
└── layer_N/
    ├── <layer_name>.pt       # QuantizedLayerData per linear
    └── non_quantized.pt      # layer norms etc.
```

## Supported Models

- `meta-llama/Llama-*`, Mistral, Qwen → `LlamaHandler`
- `facebook/opt-*` → `OPTHandler`
