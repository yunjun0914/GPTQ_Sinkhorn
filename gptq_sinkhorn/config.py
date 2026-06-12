"""Configuration dataclasses."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class QuantizationConfig:
    model_name: str
    output_dir: str
    bits: int = 4
    group_size: int = -1        # -1 = per-column scale; >0 = per-row-group
    percdamp: float = 0.01
    blocksize: int = 128
    n_samples: int = 128
    seqlen: int = 2048
    seed: int = 42
    calib_dataset: str = "c4"
    sinkhorn: bool = True       # Sinkhorn row/col normalization (comp_gh)
    hadamard_rotation: bool = False
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "bits": self.bits,
            "group_size": self.group_size,
            "percdamp": self.percdamp,
            "blocksize": self.blocksize,
            "n_samples": self.n_samples,
            "seqlen": self.seqlen,
            "seed": self.seed,
            "calib_dataset": self.calib_dataset,
            "sinkhorn": self.sinkhorn,
            "hadamard_rotation": self.hadamard_rotation,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QuantizedLayerData:
    """Quantized linear layer.

    Q      : int8  (out, in)         — symmetric quantized values in [-maxq, maxq]
    scales : fp16  (in,) or (out, n_groups)
                   group_size=-1  → per-column scale, shape (in_features,)
                   group_size>0   → per-row-group scale, shape (out, in // group_size)
    g      : fp16  (out,)            — Sinkhorn row scale
    h      : fp16  (in,)             — Sinkhorn col scale
    """
    Q: torch.Tensor
    scales: torch.Tensor
    g: torch.Tensor
    h: torch.Tensor
    bits: int = 4
    group_size: int = -1
    bias: Optional[torch.Tensor] = None
    had_d: Optional[torch.Tensor] = None     # int8 (in,) sign vector for Hadamard
    had_K_col: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "Q": self.Q,
            "scales": self.scales,
            "g": self.g,
            "h": self.h,
            "bits": self.bits,
            "group_size": self.group_size,
        }
        if self.bias is not None:
            d["bias"] = self.bias
        if self.had_d is not None:
            d["had_d"] = self.had_d
        if self.had_K_col is not None:
            d["had_K_col"] = self.had_K_col
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QuantizedLayerData":
        return cls(
            Q=d["Q"],
            scales=d["scales"],
            g=d["g"],
            h=d["h"],
            bits=int(d.get("bits", 4)),
            group_size=int(d.get("group_size", -1)),
            bias=d.get("bias"),
            had_d=d.get("had_d"),
            had_K_col=d.get("had_K_col"),
        )

    def save(self, path: Path) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "QuantizedLayerData":
        data = torch.load(path, weights_only=True, map_location="cpu")
        return cls.from_dict(data)
