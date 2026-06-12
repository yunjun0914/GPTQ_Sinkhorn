"""LLaMA / Mistral / Qwen model handler."""

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .base import ModelHandler


class LlamaHandler(ModelHandler):
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.layers

    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        layers = model.model.layers
        device = next(model.parameters()).device
        cache: Dict[str, Any] = {}
        input0_list = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                input0_list.append(inp.cpu())
                cache["attention_mask"] = kwargs.get("attention_mask")
                cache["position_ids"] = kwargs.get("position_ids")
                cache["position_embeddings"] = kwargs.get("position_embeddings")
                raise ValueError

        layers[0] = Catcher(layers[0])
        with torch.no_grad():
            for i in range(0, len(calib), batchsize):
                try:
                    model(calib[i : i + batchsize].to(device))
                except ValueError:
                    pass
        layers[0] = layers[0].module

        return torch.cat(input0_list, dim=0), cache

    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        return {
            name: mod
            for name, mod in layer.named_modules()
            if isinstance(mod, nn.Linear)
        }

    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {}
        if cache.get("position_ids") is not None:
            kwargs["position_ids"] = cache["position_ids"]
        if cache.get("position_embeddings") is not None:
            kwargs["position_embeddings"] = cache["position_embeddings"]
        return kwargs

    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        embeddings = {}
        if hasattr(model.model, "embed_tokens"):
            embeddings["model.embed_tokens"] = model.model.embed_tokens
        return embeddings

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.model.norm
