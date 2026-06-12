"""Abstract model handler interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import nn


class ModelHandler(ABC):
    @abstractmethod
    def get_layers(self, model: nn.Module) -> nn.ModuleList:
        pass

    @abstractmethod
    def get_first_layer_input(
        self, model: nn.Module, calib: torch.Tensor, batchsize: int = 8
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_linear_layers(self, layer: nn.Module) -> Dict[str, nn.Linear]:
        pass

    @abstractmethod
    def get_layer_forward_kwargs(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_embeddings(self, model: nn.Module) -> Dict[str, nn.Module]:
        pass

    @abstractmethod
    def get_final_norm(self, model: nn.Module) -> nn.Module:
        pass

    def get_num_layers(self, model: nn.Module) -> int:
        return len(self.get_layers(model))
