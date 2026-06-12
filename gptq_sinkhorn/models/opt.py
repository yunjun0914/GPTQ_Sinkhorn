"""OPT model handler."""

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .base import ModelHandler


class OPTHandler(ModelHandler):
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        return model.model.decoder.layers

    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = next(model.parameters()).device
        input0_list = []

        with torch.no_grad():
            for i in range(0, len(calib), batchsize):
                batch = calib[i : i + batchsize].to(device)
                token_embeds = model.model.decoder.embed_tokens(batch)
                attn_mask = torch.ones(batch.shape, dtype=torch.long, device=device)
                pos_embeds = model.model.decoder.embed_positions(
                    attn_mask, past_key_values_length=0
                )
                input0_list.append((token_embeds + pos_embeds).cpu())

        return torch.cat(input0_list, dim=0), {}

    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        return {
            name: mod
            for name, mod in layer.named_modules()
            if isinstance(mod, nn.Linear)
        }

    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        decoder = model.model.decoder
        embeddings = {}
        if hasattr(decoder, "embed_tokens"):
            embeddings["model.decoder.embed_tokens"] = decoder.embed_tokens
        if hasattr(decoder, "embed_positions"):
            embeddings["model.decoder.embed_positions"] = decoder.embed_positions
        return embeddings

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.model.decoder.final_layer_norm
