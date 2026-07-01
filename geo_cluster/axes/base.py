"""Segmentation-axis protocol + registry (spec-022).

An *axis* is one way to group an album (by place, time, people, content). Each
axis implements ``propose(inputs) -> Segmentation | None`` and returns ``None``
when its required signal is absent. New axes register with ``@register_axis``;
the selector discovers them via ``get_axes()`` — so adding a way to group the
album never touches the selector or scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from geo_cluster.types import GeoMetadata, Segmentation


@dataclass
class AxisInputs:
    """Every signal an axis *might* need. Axes read only what they use."""

    metadata: dict[str, GeoMetadata]
    home: Optional[tuple[float, float]] = None
    people_clusters: Optional[dict[Any, list[str]]] = None  # identity axis
    captions: Optional[dict[str, str]] = None               # semantic axis
    config: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SegmentationAxis(Protocol):
    name: str
    needs: str  # human-readable description of the required signal

    def propose(self, inputs: AxisInputs) -> Optional[Segmentation]:
        ...

    def perturbed_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a slightly-nudged config, used to measure stability."""
        ...


_AXES: dict[str, SegmentationAxis] = {}


def register_axis(cls):
    """Class decorator: instantiate and register an axis by its ``name``."""
    inst = cls()
    _AXES[inst.name] = inst
    return cls


def get_axes() -> dict[str, SegmentationAxis]:
    return dict(_AXES)
