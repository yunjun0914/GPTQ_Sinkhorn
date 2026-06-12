"""3D surface visualization of weight matrices before GPTQ.

Finds the linear sublayer with the highest baseline std, then plots
a 2×3 grid showing each transform step:

  Row 1  gh → had :  W_orig  |  W/g/h       |  had(W/g/h)
  Row 2  had → gh :  W_orig  |  had(W)      |  had(W)/g'/h'

Usage:
    # auto-detect layer with max baseline std
    python visualize_weight_transforms.py /path/to/llama2-7b

    # specify manually
    python visualize_weight_transforms.py /path/to/llama2-7b \
        --layer 15 --sublayer self_attn.q_proj

    # control subsampling (default stride=16 → 256×256 surface)
    python visualize_weight_transforms.py /path/to/llama2-7b --stride 32

    # save to custom path
    python visualize_weight_transforms.py /path/to/llama2-7b --out weight_viz.png
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")          # headless-safe; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from transformers import AutoModelForCausalLM

from gptq_sinkhorn.algorithms.gptq import comp_gh
from gptq_sinkhorn.models import detect_and_get_handler
from gptq_sinkhorn.utils.hadamard import apply_had_to_W_single, get_hadK, make_had_d


# ─────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def _had_params(in_dim: int, device):
    had_K, had_K_col = get_hadK(in_dim)
    had_d = make_had_d(in_dim, device, seed=0)
    return had_K, had_K_col, had_d


def make_stages_gh_then_had(W: torch.Tensor):
    """Return (W_orig, W_after_gh, W_after_had) for gh → had pipeline."""
    W = W.float()
    had_K, had_K_col, had_d = _had_params(W.shape[1], W.device)

    g, h = comp_gh(W)
    W_gh = W / g[:, None] / h[None, :]
    W_had = apply_had_to_W_single(W_gh, had_d, had_K, had_K_col)

    return W, W_gh, W_had


def make_stages_had_then_gh(W: torch.Tensor):
    """Return (W_orig, W_after_had, W_after_gh) for had → gh pipeline."""
    W = W.float()
    had_K, had_K_col, had_d = _had_params(W.shape[1], W.device)

    W_had = apply_had_to_W_single(W, had_d, had_K, had_K_col)
    g, h = comp_gh(W_had)
    W_gh = W_had / g[:, None] / h[None, :]

    return W, W_had, W_gh


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _stats_str(W: torch.Tensor) -> str:
    flat = W.float().flatten()
    std = flat.std().item()
    max_abs = flat.abs().max().item()
    mu = flat.mean()
    sigma = flat.std().clamp(min=1e-8)
    kurt = (((flat - mu) / sigma) ** 4).mean().item() - 3.0
    return f"std={std:.4f}  max|w|={max_abs:.4f}  kurt={kurt:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# 3D surface helper
# ─────────────────────────────────────────────────────────────────────────────

def _surface(ax, W: torch.Tensor, stride: int, title: str, subtitle: str,
             elev: int = 30, azim: int = -60):
    """Plot W[::stride, ::stride] as a 3D surface on ax."""
    Z = W.float().numpy()[::stride, ::stride]
    n_out, n_in = Z.shape

    X, Y = np.meshgrid(np.arange(n_in), np.arange(n_out))

    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           linewidth=0, antialiased=True)

    ax.set_xlabel("Input Channel", fontsize=7, labelpad=4)
    ax.set_ylabel("Output Channel", fontsize=7, labelpad=4)
    ax.set_zlabel("Value", fontsize=7, labelpad=2)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.text2D(0.5, -0.04, subtitle, transform=ax.transAxes,
              ha="center", fontsize=6.5, color="#444444")

    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=6)

    return surf


# ─────────────────────────────────────────────────────────────────────────────
# Auto-find layer with max baseline std
# ─────────────────────────────────────────────────────────────────────────────

def find_max_kurtosis_layer(model, handler):
    layers = handler.get_layers(model)
    best = dict(kurtosis=-999.0, layer_idx=-1, name="")

    for layer_idx, layer in enumerate(layers):
        for name, linear in handler.get_linear_layers(layer).items():
            W = linear.weight.data.float()
            flat = W.flatten()
            mu = flat.mean()
            sigma = flat.std().clamp(min=1e-8)
            kurt = (((flat - mu) / sigma) ** 4).mean().item() - 3.0
            if kurt > best["kurtosis"]:
                best = dict(kurtosis=kurt, layer_idx=layer_idx, name=name)
        print(f"  scanned layer {layer_idx:2d} ...", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    return best["layer_idx"], best["name"], best["kurtosis"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--layer",    type=int,   default=None, help="Layer index (auto if omitted)")
    p.add_argument("--sublayer", type=str,   default=None, help="Sublayer name e.g. self_attn.q_proj")
    p.add_argument("--stride",   type=int,   default=16,   help="Subsample stride (default 16 → 256×256)")
    p.add_argument("--elev",     type=int,   default=30)
    p.add_argument("--azim",     type=int,   default=-60)
    p.add_argument("--out",      type=str,   default="weight_transform_viz.png")
    args = p.parse_args()

    print(f"Loading model {args.model_path} ...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    handler = detect_and_get_handler(model)
    layers  = handler.get_layers(model)

    # ── Select target layer ────────────────────────────────────────────────
    if args.layer is None or args.sublayer is None:
        print("Scanning layers for max baseline kurtosis ...", file=sys.stderr)
        layer_idx, sublayer_name, max_kurt = find_max_kurtosis_layer(model, handler)
        print(f"  → layer {layer_idx} / {sublayer_name}  (kurtosis={max_kurt:.4f})", file=sys.stderr)
        if args.layer is not None:
            layer_idx = args.layer
        if args.sublayer is not None:
            sublayer_name = args.sublayer
    else:
        layer_idx    = args.layer
        sublayer_name = args.sublayer

    target_layer   = layers[layer_idx]
    linear_layers  = handler.get_linear_layers(target_layer)
    assert sublayer_name in linear_layers, \
        f"Sublayer '{sublayer_name}' not found. Available: {list(linear_layers)}"

    W = linear_layers[sublayer_name].weight.data.float()
    print(f"Target: layer_{layer_idx}.{sublayer_name}  shape={tuple(W.shape)}", file=sys.stderr)

    # ── Compute transform stages ───────────────────────────────────────────
    print("Applying transforms ...", file=sys.stderr)
    gh_orig, gh_mid, gh_final   = make_stages_gh_then_had(W)   # gh → had
    hg_orig, hg_mid, hg_final   = make_stages_had_then_gh(W)   # had → gh

    label = f"layer_{layer_idx}.{sublayer_name}"
    stride = args.stride
    n_out_vis = W.shape[0] // stride
    n_in_vis  = W.shape[1] // stride
    print(f"Surface grid: {n_out_vis}×{n_in_vis}  (stride={stride})", file=sys.stderr)

    # ── Figure layout: 2 rows × 3 cols ────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Weight Distribution Before GPTQ  —  {label}\n"
        f"stride={stride} ({n_out_vis}×{n_in_vis} shown out of {W.shape[0]}×{W.shape[1]})",
        fontsize=11, fontweight="bold", y=0.98
    )

    row_labels = ["gh → had", "had → gh"]
    col_labels_gh  = ["Original W", "After Sinkhorn (W/g/h)", "After Hadamard  had(W/g/h)"]
    col_labels_hg  = ["Original W", "After Hadamard  had(W)", "After Sinkhorn  had(W)/g'/h'"]

    panels = [
        # (row, stages_list, col_titles)
        (0, [gh_orig, gh_mid, gh_final], col_labels_gh),
        (1, [hg_orig, hg_mid, hg_final], col_labels_hg),
    ]

    for row_idx, (row, stages, col_labels) in enumerate(panels):
        for col_idx, (stage_W, col_label) in enumerate(zip(stages, col_labels)):
            ax_idx = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(2, 3, ax_idx, projection="3d")

            title    = col_label
            subtitle = _stats_str(stage_W)

            _surface(ax, stage_W, stride, title, subtitle,
                     elev=args.elev, azim=args.azim)

        # Row label on the leftmost subplot
        left_ax_idx = row_idx * 3 + 1
        left_ax = fig.axes[left_ax_idx - 1]
        left_ax.text2D(-0.12, 0.5, row_labels[row_idx],
                       transform=left_ax.transAxes,
                       fontsize=11, fontweight="bold", color="#222222",
                       rotation=90, va="center", ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
