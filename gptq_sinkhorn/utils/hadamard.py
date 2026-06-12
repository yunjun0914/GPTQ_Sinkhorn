"""Random Hadamard Transform utilities (ported from eptq-idea)."""

import math
import os
from typing import Optional, Tuple

import torch

try:
    from fast_hadamard_transform import hadamard_transform as _fht_cuda
    _HAS_FAST_HAD = True
except ImportError:
    _HAS_FAST_HAD = False

_USE_FHT: bool = os.environ.get("GPTQ_USE_FHT", "1" if _HAS_FAST_HAD else "0") == "1"
_FHT_BATCH_THRESHOLD: int = int(os.environ.get("GPTQ_FHT_BATCH_THRESHOLD", "4"))

_had_cache: dict = {}
_had_gpu_cache: dict = {}


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def get_block_size(n: int) -> int:
    """Largest power-of-2 that divides n."""
    if is_power_of_2(n):
        return n
    k = 1
    while k * 2 <= n and n % (k * 2) == 0:
        k *= 2
    return k


def get_hadamard(n: int) -> torch.Tensor:
    """Normalized n×n Hadamard matrix on CPU. n must be power-of-2."""
    assert is_power_of_2(n), f"Hadamard requires power-of-2, got {n}"
    if n not in _had_cache:
        H = torch.ones(1, 1, dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _had_cache[n] = H / math.sqrt(n)
    return _had_cache[n]


def apply_hadamard(X: torch.Tensor) -> torch.Tensor:
    """Block Hadamard on last dim via dense matmul (self-inverse)."""
    n = X.shape[-1]
    orig_shape = X.shape
    block = get_block_size(n)

    if X.is_cuda:
        key = (block, X.device.index)
        if key not in _had_gpu_cache:
            _had_gpu_cache[key] = get_hadamard(block).to(X.device)
        H = _had_gpu_cache[key]
    else:
        H = get_hadamard(block).to(X.device)

    if block == n:
        return (X.float().reshape(-1, n) @ H).reshape(orig_shape).to(X.dtype)
    else:
        n_blocks = n // block
        x = X.float().reshape(*orig_shape[:-1], n_blocks, block)
        return (x @ H).reshape(orig_shape).to(X.dtype)


def get_hadK(n: int) -> Tuple[Optional[torch.Tensor], int]:
    """Return (hadK=None, K) for dimension n. K = number of blocks."""
    if is_power_of_2(n):
        return None, 1
    block = get_block_size(n)
    return None, n // block


def matmul_hadU(X: torch.Tensor, hadK: Optional[torch.Tensor], K: int) -> torch.Tensor:
    if _USE_FHT and K == 1:
        n_rows = X.reshape(-1, X.shape[-1]).shape[0]
        if n_rows <= _FHT_BATCH_THRESHOLD:
            n = X.shape[-1]
            orig_shape = X.shape
            return fast_hadamard_transform(X.reshape(-1, n).float()).reshape(orig_shape).to(X.dtype)
    return apply_hadamard(X)


def matmul_hadUt(X: torch.Tensor, hadK: Optional[torch.Tensor], K: int) -> torch.Tensor:
    return matmul_hadU(X, hadK, K)


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform. Last dim must be power-of-2."""
    n = x.shape[-1]
    assert is_power_of_2(n), f"last dim must be power of 2, got {n}"
    if _HAS_FAST_HAD and x.is_cuda:
        return _fht_cuda(x.contiguous(), scale=1.0 / math.sqrt(n))
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], -1, 2 * h)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1)
        x = x.reshape(*x.shape[:-2], -1)
        h *= 2
    return x / math.sqrt(n)


def make_had_d(n: int, device, seed: int = 0) -> torch.Tensor:
    """Random ±1 sign vector of length n."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    d = torch.randint(0, 2, (n,), generator=rng, dtype=torch.int8) * 2 - 1
    return d.to(device)


def apply_had_to_W_single(
    W: torch.Tensor, d: torch.Tensor, hadK: Optional[torch.Tensor], K: int
) -> torch.Tensor:
    """Right rotation: W' = W @ diag(d) @ U.  W: (out, in)."""
    return matmul_hadU(W * d.to(W.device).float()[None, :], hadK, K).to(W.dtype)


def apply_inverse_had_to_W_single(
    W: torch.Tensor, d: torch.Tensor, hadK: Optional[torch.Tensor], K: int
) -> torch.Tensor:
    """Undo apply_had_to_W_single. W: (out, in)."""
    x = matmul_hadUt(W.float(), hadK, K)
    x = x * d.to(W.device).float()[None, :]
    return x.to(W.dtype)


def apply_had_to_H_single(
    H: torch.Tensor, d: torch.Tensor, hadK: Optional[torch.Tensor], K: int
) -> torch.Tensor:
    """H_rot = V^T @ H @ V for V = diag(d) @ U."""
    H1 = apply_had_to_W_single(H, d, hadK, K)
    return apply_had_to_W_single(H1.T.contiguous(), d, hadK, K).T
