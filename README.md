# GPTQ-Sinkhorn

GPTQ error propagation + symmetric RTN quantization with **Sinkhorn normalization** and **randomized Hadamard rotation** for LLM post-training quantization.

## Key Ideas

| Component | Description |
|-----------|-------------|
| **Sinkhorn normalization** | Iterative row/col scaling (`comp_gh`) that balances weight magnitude before quantization |
| **Hadamard rotation** | Randomized block-Hadamard transform applied to weights for incoherence processing |
| **GPTQ** | Column-wise quantization with error propagation (Cholesky-based) |
| **Symmetric RTN** | Simple integer quantization: `Q = round(W / scale).clamp(-maxq, maxq)` |

### Pipeline (per linear layer)

```
W, H = 2·X^T·X / N
  → comp_gh(W)            [--sinkhorn]          Sinkhorn → g, h
  → W /= g·h,  H *= h²
  → Hadamard(W, H)        [--hadamard_rotation] incoherence
  → GPTQ + sym RTN                              column-wise quantize + error propagation
```

## Installation

```bash
git clone https://github.com/yunjun0914/GPTQ_Sinkhorn.git
cd GPTQ_Sinkhorn
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, transformers, datasets

## Usage

### 1. Quantization

```bash
python quantize.py <model_name_or_path> --output_dir <output_dir> [options]
```

**Recommended config (Llama-2-7b):**

```bash
python quantize.py meta-llama/Llama-2-7b-hf \
    --output_dir ./quantized_llama2-7b \
    --bits 4 \
    --sinkhorn \
    --hadamard_rotation
```

**Full options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--bits` | `4` | Quantization bits (2/3/4/8) |
| `--group_size` | `-1` | Columns per group. `-1` = per-column scale (recommended with Sinkhorn). `128` = per-row-group scale |
| `--sinkhorn` / `--no_sinkhorn` | on | Sinkhorn row/col normalization before quantization |
| `--hadamard_rotation` | off | Randomized Hadamard rotation for incoherence |
| `--percdamp` | `0.01` | Hessian damping ratio |
| `--blocksize` | `128` | GPTQ block size |
| `--n_samples` | `128` | Number of calibration samples |
| `--seqlen` | `2048` | Calibration sequence length |
| `--calib_dataset` | `c4` | Calibration dataset (`c4` or `wikitext2`) |
| `--device` | `cuda` | Device to run on |

### 2. Reconstruction (dequantize → fp16)

```bash
python reconstruct.py \
    --quantized_dir ./quantized_llama2-7b \
    --output_dir ./reconstructed_llama2-7b
```

Produces a standard HuggingFace model directory that can be loaded with `from_pretrained`.

### 3. PPL Evaluation

```bash
# Evaluate reconstructed model
python eval_ppl.py --model_dir ./reconstructed_llama2-7b --dataset wikitext2 c4

# Evaluate baseline (fp16)
python eval_ppl.py --model_name meta-llama/Llama-2-7b-hf --dataset wikitext2 c4
```

## Output Format

```
quantized_model/
├── quant_config.json          # QuantizationConfig
├── global_params.pt           # embeddings, final norm, lm_head
├── tokenizer/
└── layer_N/
    ├── <layer_name>.pt        # QuantizedLayerData per linear layer
    └── non_quantized.pt       # layer norms etc.
```

**QuantizedLayerData fields:**

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `Q` | int8 | (out, in) | Quantized indices in `[-maxq, maxq]` |
| `scales` | fp16 | (in,) or (out, n_groups) | Per-column or per-row-group scale |
| `g` | fp16 | (out,) | Sinkhorn row scale |
| `h` | fp16 | (in,) | Sinkhorn col scale |
| `had_d` | int8 | (in,) | Hadamard ±1 sign vector |
| `had_K_col` | int | scalar | Number of Hadamard blocks |

## Supported Models

- `meta-llama/Llama-2-*`, `meta-llama/Llama-3-*`
- `mistralai/Mistral-*`
- `Qwen/Qwen*`
- `facebook/opt-*`

## Project Structure

```
GPTQ_Sinkhorn/
├── gptq_sinkhorn/
│   ├── config.py              # QuantizationConfig, QuantizedLayerData
│   ├── algorithms/
│   │   └── gptq.py            # comp_gh (Sinkhorn), gptq_quantize, dequantize_layer
│   ├── models/
│   │   ├── llama.py           # LlamaHandler (Llama, Mistral, Qwen)
│   │   └── opt.py             # OPTHandler
│   └── utils/
│       ├── hadamard.py        # Block-Hadamard transform utilities
│       ├── hessian.py         # Hessian accumulation via forward hooks
│       └── calibration.py     # C4 / wikitext2 calibration data
├── quantize.py                # Quantization entry point
├── reconstruct.py             # Dequantize to fp16
└── eval_ppl.py                # PPL evaluation (wikitext2, c4)
```

## License

MIT
