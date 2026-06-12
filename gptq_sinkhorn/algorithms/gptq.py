"""GPTQ quantization with Sinkhorn normalization and Hadamard rotation."""

import sys
from typing import Optional, Tuple

import torch

from ..config import QuantizedLayerData


def comp_gh(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn row/column normalization.

    Iteratively normalizes so that mean(|W_norm[i,:]|) ≈ 1 per row and column.
    Returns (g, h) where W_norm = W / g[:, None] / h[None, :].
    """
    Wabs = W.abs()
    g = torch.ones(Wabs.shape[0], device=Wabs.device)
    h = torch.ones(Wabs.shape[1], device=Wabs.device)
    for _ in range(10):
        Wabs_h = Wabs / h[None, :].clamp(min=1e-8)
        g = Wabs_h.mean(dim=1).clamp(min=1e-8)
        Wabs_g = Wabs / g[:, None].clamp(min=1e-8)
        h = Wabs_g.mean(dim=0).clamp(min=1e-8)
    return g, h


def _column_wise_gptq(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    bits: int,
    blocksize: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ column-wise quantization with symmetric RTN.

    group_size=-1 : per-column scale — one scalar per column, shared across all rows.
                    scales shape: (in_features,)
    group_size>0  : per-row-group scale — each row has its own scale per group of columns.
                    Scales are pre-computed from the normalized W before GPTQ.
                    scales shape: (out_features, n_groups)

    Returns Q (int8), scales (fp32), W_q (fp32 dequantized).
    Mutates W in-place via error propagation.
    """
    out_dim, in_dim = W.shape
    maxq = 2 ** (bits - 1) - 1

    Q = torch.zeros((out_dim, in_dim), dtype=torch.int8, device=W.device)
    W_q = torch.zeros_like(W)

    if group_size == -1:
        scales = torch.zeros(in_dim, device=W.device, dtype=torch.float32)
    else:
        n_groups = (in_dim + group_size - 1) // group_size
        scales = torch.zeros(out_dim, n_groups, device=W.device, dtype=torch.float32)
        # Pre-compute per-row-group scales from current W state
        for k in range(n_groups):
            g_start = k * group_size
            g_end = min(g_start + group_size, in_dim)
            scales[:, k] = W[:, g_start:g_end].abs().amax(dim=1).clamp(min=1e-8) / maxq

    for i in range(0, in_dim, blocksize):
        i_end = min(i + blocksize, in_dim)
        W1 = W[:, i:i_end].clone()
        E1 = torch.zeros_like(W1)
        H_inv1 = H_inv[i:i_end, i:i_end]

        for j in range(i_end - i):
            col_idx = i + j
            col = W1[:, j]

            if group_size == -1:
                scale = col.abs().max().clamp(min=1e-8) / maxq
                scales[col_idx] = scale
                q = (col / scale).round().clamp(-maxq, maxq)
                wq = q * scale
            else:
                gk = col_idx // group_size
                scale = scales[:, gk]           # (out_dim,)
                q = (col / scale).round().clamp(-maxq, maxq)
                wq = q * scale

            Q[:, col_idx] = q.to(torch.int8)
            W_q[:, col_idx] = wq

            d = H_inv1[j, j]
            E1[:, j] = (col - wq) / d
            W1[:, j:] -= torch.outer(E1[:, j], H_inv1[j, j:])

        W[:, i_end:] -= E1 @ H_inv[i:i_end, i_end:]

    return Q, scales, W_q


def _compute_H_inv(
    H: torch.Tensor,
    dead: torch.Tensor,
    h: torch.Tensor,
    percdamp: float,
    sinkhorn: bool,
    layer_name: str,
) -> torch.Tensor:
    """Compute upper-triangular Cholesky factor of H^{-1} with damping retry."""
    device = H.device
    percdamp_curr = percdamp

    for attempt in range(60):
        try:
            H_work = H.clone()
            damp = percdamp_curr * H_work.diagonal().mean()
            H_work.diagonal().add_(damp)
            L = torch.linalg.cholesky(H_work)
            H_inv_full = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(H_inv_full, upper=True)
        except torch._C._LinAlgError:
            percdamp_curr *= 2
            if attempt == 59:
                print(
                    f"Cholesky failed after 60 retries for {layer_name}; using identity.",
                    file=sys.stderr,
                )
                return torch.eye(H.shape[0], device=device) * 0.01
    # unreachable
    return torch.eye(H.shape[0], device=device) * 0.01


def gptq_quantize(
    H_orig: torch.Tensor,
    W_orig: torch.Tensor,
    bits: int = 4,
    percdamp: float = 0.01,
    blocksize: int = 128,
    group_size: int = -1,
    sinkhorn: bool = True,
    hadamard_rotation: bool = False,
    layer_name: str = "",
) -> QuantizedLayerData:
    """GPTQ quantization with optional Sinkhorn normalization and Hadamard rotation.

    Pipeline:
        W, H
        → comp_gh(W)                [if sinkhorn]   Sinkhorn → g, h
        → W /= g·h,  H *= h²
        → Hadamard rotation(W, H)   [if hadamard]   incoherence
        → GPTQ column-wise + sym RTN
    """
    device = W_orig.device
    H = H_orig.detach().clone().float()
    W = W_orig.detach().clone().float()

    # Sinkhorn normalization
    if sinkhorn:
        g, h = comp_gh(W)
        W = W / g[:, None] / h[None, :]
        H = H * h[:, None] * h[None, :]
    else:
        g = torch.ones(W.shape[0], device=device)
        h = torch.ones(W.shape[1], device=device)

    # Dead weight handling (zero diagonal in H → column never activated)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    had_d = None
    had_K_col = None

    if hadamard_rotation:
        from ..utils.hadamard import (
            apply_had_to_H_single,
            apply_had_to_W_single,
            get_hadK,
            make_had_d,
        )
        hadK_col, had_K_col = get_hadK(W.shape[1])
        had_d = make_had_d(W.shape[1], device, seed=0)
        W = apply_had_to_W_single(W, had_d, hadK_col, had_K_col)
        H = apply_had_to_H_single(H, had_d, hadK_col, had_K_col)

    H_inv = _compute_H_inv(H, dead, h, percdamp, sinkhorn, layer_name)

    Q, scales, _ = _column_wise_gptq(W, H_inv, bits, blocksize, group_size)

    return QuantizedLayerData(
        Q=Q,
        scales=scales.half(),
        g=g.half(),
        h=h.half(),
        bits=bits,
        group_size=group_size,
        bias=None,
        had_d=had_d.cpu() if had_d is not None else None,
        had_K_col=had_K_col,
    )


def dequantize_layer(quant_data: QuantizedLayerData) -> torch.Tensor:
    """Reconstruct fp16 weights from QuantizedLayerData.

    Inverse of gptq_quantize:
        W_q = Q * scales
        → undo Hadamard  [if had_d is set]
        → W = W_q * g * h
    """
    Q = quant_data.Q.float()
    scales = quant_data.scales.float()
    out_dim, in_dim = Q.shape

    if quant_data.group_size == -1:
        W_q = Q * scales[None, :]
    else:
        group_size = quant_data.group_size
        n_groups = scales.shape[1]
        W_q = torch.zeros_like(Q)
        for k in range(n_groups):
            g_start = k * group_size
            g_end = min(g_start + group_size, in_dim)
            W_q[:, g_start:g_end] = Q[:, g_start:g_end] * scales[:, k : k + 1]

    if quant_data.had_d is not None:
        from ..utils.hadamard import apply_inverse_had_to_W_single, get_hadK

        device = W_q.device
        had_d = quant_data.had_d.to(device)
        hadK_col, _ = get_hadK(in_dim)
        W_q = apply_inverse_had_to_W_single(W_q, had_d, hadK_col, quant_data.had_K_col)

    g = quant_data.g.float().to(W_q.device)
    h = quant_data.h.float().to(W_q.device)
    W = W_q * g[:, None] * h[None, :]

    return W.half()
