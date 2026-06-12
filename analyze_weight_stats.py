"""Weight distribution analysis before GPTQ under different transform orderings.

Compares W statistics at the point just before GPTQ for:
  baseline   : W (no transforms)
  sinkhorn   : W / g / h  (Sinkhorn only)
  gh_then_had: had(W/g/h) (Sinkhorn → Hadamard, default pipeline)
  had_then_gh: had(W)/g'/h'  (Hadamard → Sinkhorn, rotation_first)

Theory: gh_then_had first removes row/col outliers via Sinkhorn, then rotates an
already-normalized W. By CLT, the result is near-Gaussian (low kurtosis).
had_then_gh rotates an outlier-y W first; heavy-tail energy is spread but not
removed before Sinkhorn, so kurtosis stays higher.

Usage:
    python analyze_weight_stats.py /path/to/llama2-7b
    python analyze_weight_stats.py /path/to/llama2-7b --layers 0 1 2   # subset
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM

from gptq_sinkhorn.algorithms.gptq import comp_gh
from gptq_sinkhorn.models import detect_and_get_handler
from gptq_sinkhorn.utils.hadamard import apply_had_to_W_single, get_hadK, make_had_d


MODES = ["baseline", "sinkhorn", "gh_then_had", "had_then_gh"]
MODE_LABELS = {
    "baseline":    "baseline   ",
    "sinkhorn":    "sinkhorn   ",
    "gh_then_had": "gh→had     ",
    "had_then_gh": "had→gh     ",
}


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _excess_kurtosis(x: torch.Tensor) -> float:
    x = x.float().flatten()
    mu = x.mean()
    sigma = x.std().clamp(min=1e-8)
    return (((x - mu) / sigma) ** 4).mean().item() - 3.0


def weight_stats(W: torch.Tensor) -> dict:
    """Compute distribution statistics for weight matrix W (out, in)."""
    W = W.float()
    flat = W.flatten()

    std      = flat.std().item()
    max_abs  = flat.abs().max().item()
    kurt     = _excess_kurtosis(flat)

    col_max   = W.abs().amax(dim=0)            # max per input column  (in,)
    row_max   = W.abs().amax(dim=1)            # max per output row    (out,)

    # Coefficient of variation of per-column max: std/mean (lower = more uniform columns)
    col_max_cv  = (col_max.std() / col_max.mean().clamp(min=1e-8)).item()
    col_max_ratio = (col_max.max() / col_max.min().clamp(min=1e-8)).item()

    # Same for rows
    row_max_cv   = (row_max.std() / row_max.mean().clamp(min=1e-8)).item()

    return dict(
        std=std,
        max_abs=max_abs,
        kurtosis=kurt,
        col_max_cv=col_max_cv,
        col_max_ratio=col_max_ratio,
        row_max_cv=row_max_cv,
    )


def apply_transforms(W: torch.Tensor, mode: str,
                     had_d: torch.Tensor,
                     had_K: object, had_K_col: int) -> torch.Tensor:
    W = W.float()
    if mode == "baseline":
        return W
    if mode == "sinkhorn":
        g, h = comp_gh(W)
        return W / g[:, None] / h[None, :]
    if mode == "gh_then_had":
        g, h = comp_gh(W)
        W_norm = W / g[:, None] / h[None, :]
        return apply_had_to_W_single(W_norm, had_d, had_K, had_K_col)
    if mode == "had_then_gh":
        W_rot = apply_had_to_W_single(W, had_d, had_K, had_K_col)
        g, h = comp_gh(W_rot)
        return W_rot / g[:, None] / h[None, :]
    raise ValueError(f"Unknown mode: {mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ──────────────────────────────────────────────────────────────────────────────

STAT_KEYS = ["std", "max_abs", "kurtosis", "col_max_cv", "col_max_ratio", "row_max_cv"]
HEADER_FIELDS = ["std", "max_abs", "kurtosis", "col_max_cv", "col_max_ratio", "row_max_cv"]
COL_W = dict(std=8, max_abs=8, kurtosis=10, col_max_cv=11, col_max_ratio=13, row_max_cv=11)

def _header_line(label_w: int = 40) -> str:
    cols = "  ".join(f"{k:>{COL_W[k]}}" for k in HEADER_FIELDS)
    return f"{'name':<{label_w}} {'mode':<12} {cols}"

def _data_line(name: str, mode: str, stats: dict, label_w: int = 40) -> str:
    vals = "  ".join(
        f"{stats[k]:>{COL_W[k]}.4f}" if k not in ("col_max_ratio",)
        else f"{stats[k]:>{COL_W[k]}.1f}"
        for k in HEADER_FIELDS
    )
    return f"{name:<{label_w}} {MODE_LABELS[mode]:<12} {vals}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Layer indices to analyse (default: all)")
    args = p.parse_args()

    print(f"Loading model {args.model_path} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()

    handler  = detect_and_get_handler(model)
    layers   = handler.get_layers(model)
    n_layers = len(layers)

    layer_indices = args.layers if args.layers is not None else list(range(n_layers))

    # ── Accumulators ──────────────────────────────────────────────────────────
    agg = {m: {k: [] for k in STAT_KEYS} for m in MODES}

    # ── Print header ──────────────────────────────────────────────────────────
    sep = "─" * 120
    print(sep)
    print(_header_line())
    print(sep)

    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        linear_layers = handler.get_linear_layers(layer)

        for name, linear in linear_layers.items():
            W = linear.weight.data.float()
            in_dim = W.shape[1]

            had_K, had_K_col = get_hadK(in_dim)
            had_d = make_had_d(in_dim, W.device, seed=0)

            full_name = f"layer_{layer_idx:02d}.{name}"

            for mode in MODES:
                W_t = apply_transforms(W, mode, had_d, had_K, had_K_col)
                stats = weight_stats(W_t)

                for k in STAT_KEYS:
                    agg[mode][k].append(stats[k])

                print(_data_line(full_name, mode, stats))

            print()  # blank line between sublayers

    # ── Summary: average across all layers ────────────────────────────────────
    print(sep)
    print("AVERAGE ACROSS ALL LAYERS")
    print(sep)
    print(_header_line())
    print(sep)

    avg_all = {}
    for mode in MODES:
        avg = {k: sum(agg[mode][k]) / len(agg[mode][k]) for k in STAT_KEYS}
        avg_all[mode] = avg
        print(_data_line("ALL_LAYERS_AVG", mode, avg))

    # ── Delta table: gh_then_had vs had_then_gh ───────────────────────────────
    print()
    print(sep)
    print("gh→had  vs  had→gh  (ratio or diff, lower is better for all)")
    print(sep)
    print(f"{'metric':<20}  {'gh→had':>12}  {'had→gh':>12}  {'gh→had / had→gh':>18}  {'better':>10}")
    print(sep)
    for k in STAT_KEYS:
        a = avg_all["gh_then_had"][k]
        b = avg_all["had_then_gh"][k]
        ratio = a / b if b != 0 else float("inf")
        better = "gh→had" if a < b else ("had→gh" if b < a else "tie")
        print(f"{k:<20}  {a:>12.4f}  {b:>12.4f}  {ratio:>18.4f}  {better:>10}")


if __name__ == "__main__":
    main()
