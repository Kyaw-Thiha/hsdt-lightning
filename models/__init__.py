from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Tuple

import torch.nn as nn

from .hsdt import HSDT
from .ssrt_unet.ssrt import ssrt as SSRT
from .hdst.sert import SERT
from .tdsat.tdsat import TDSAT


ModelFactory = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    """Container describing how to instantiate a model and its default arguments."""

    factory: ModelFactory
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def resolve_kwargs(self, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Merge default kwargs with any overrides supplied by the caller."""
        params: Dict[str, Any] = dict(self.default_kwargs)
        if overrides:
            params.update(overrides)
        return params

    def instantiate(self, overrides: Mapping[str, Any] | None = None) -> nn.Module:
        """Create a model instance using merged kwargs."""
        return self.factory(**self.resolve_kwargs(overrides))


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "hsdt": ModelSpec(
        factory=HSDT,
        default_kwargs={
            "in_channels": 1,
            "channels": 16,
            "num_half_layer": 5,
            "downsample_layers": (1, 3),
            "num_bands": 81,
        },
    ),
    "ssrt": ModelSpec(
        factory=SSRT,
        default_kwargs={
            "upscale": 1,
            "img_size": (64, 64),
            "window_size": 8,
            "img_range": 1.0,
            "depths": (2, 2, 6, 2, 2),
            "embed_dim": 16,
            "num_heads": (2, 2, 2, 2, 2),
            "mlp_ratio": 2,
            "upsampler": None,
            "in_chans": 1,
            "gate": "sru",
            "if_mlp_s": True,
        },
    ),
    "hdst": ModelSpec(
        factory=SERT,
        default_kwargs={
            "inp_channels": 81,
            "dim": 64,
            "window_sizes": (16, 32, 32),
            "depths": (6, 6, 6),
            "num_heads": (4, 4, 4),
            "split_sizes": (1, 2, 4),
            "mlp_ratio": 2,
            "weight_factor": 0.1,
            "memory_blocks": 128,
            "down_rank": 8,
        },
    ),
    "tdsat": ModelSpec(
        factory=TDSAT,
        default_kwargs={
            "in_channels": 1,
            "channels": 16,
            "num_half_layer": 5,
            "sample_idx": (1, 3),
        },
    ),
}


def available_models() -> Tuple[str, ...]:
    """Return the tuple of supported model names."""
    return tuple(MODEL_REGISTRY.keys())


def get_model_spec(name: str) -> ModelSpec:
    """Retrieve the model specification for a given registry name."""
    normalized = name.lower()
    if normalized not in MODEL_REGISTRY:
        options = ", ".join(available_models())
        raise KeyError(f"Unknown model '{name}'. Available options: {options}")
    return MODEL_REGISTRY[normalized]


def build_model(name: str, **overrides: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate a model by name and return it along with the resolved kwargs."""
    spec = get_model_spec(name)
    resolved = spec.resolve_kwargs(overrides)
    return spec.factory(**resolved), resolved


__all__ = [
    "HSDT",
    "SSRT",
    "SERT",
    "TDSAT",
    "ModelSpec",
    "MODEL_REGISTRY",
    "available_models",
    "get_model_spec",
    "build_model",
]
