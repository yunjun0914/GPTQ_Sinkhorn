"""Hessian computation via forward hooks."""

from typing import Any, Dict, Optional

import torch
from torch import nn


@torch.no_grad()
def get_hessians(
    layer: nn.Module,
    layer_input: torch.Tensor,
    batchsize: int = 4,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute H = X^T X / (N/2) for every nn.Linear in layer."""
    if forward_kwargs is None:
        forward_kwargs = {}

    device = next(layer.parameters()).device
    n_samples = layer_input.shape[0]
    hnorm = torch.tensor(n_samples / 2.0, device=device)

    linear_layers: Dict[str, nn.Linear] = {
        name: mod
        for name, mod in layer.named_modules()
        if isinstance(mod, nn.Linear)
    }

    hessians: Dict[str, torch.Tensor] = {
        name: torch.zeros(mod.in_features, mod.in_features, device=device, dtype=torch.float32)
        for name, mod in linear_layers.items()
    }

    def make_hook(mod_name: str):
        def hook(module, inputs, output):
            x = inputs[0].view(-1, inputs[0].shape[-1]).float()
            hessians[mod_name] += (x.T @ x) / hnorm
        return hook

    hooks = [mod.register_forward_hook(make_hook(name)) for name, mod in linear_layers.items()]

    for i in range(0, n_samples, batchsize):
        batch = layer_input[i : i + batchsize].to(device)
        layer(batch, **forward_kwargs)

    for h in hooks:
        h.remove()

    return hessians


@torch.no_grad()
def get_layer_output(
    layer: nn.Module,
    layer_input: torch.Tensor,
    batchsize: int = 4,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Run layer forward pass and collect outputs."""
    if forward_kwargs is None:
        forward_kwargs = {}

    device = next(layer.parameters()).device
    outputs = []

    for i in range(0, layer_input.shape[0], batchsize):
        batch = layer_input[i : i + batchsize].to(device)
        out = layer(batch, **forward_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        outputs.append(out.detach().cpu())

    return torch.cat(outputs, dim=0)
