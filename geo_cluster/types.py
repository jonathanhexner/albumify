"""Geo-temporal types (spec-022).

Framework-agnostic dataclasses shared by the pipeline steps and notebook
callers. No dependency on PipelineContext — the step is the translator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SegmentKind(str, Enum):
    """What a segment *is* — assigned after the grouping axis is chosen.

    The geo/time axes use TRIP/DAY_OUT/EVENT/EVERYDAY; the identity axis uses
    PERSON; the semantic axis uses THEME. UNSORTED is the no-signal bucket.
    """

    TRIP = "trip"
    DAY_OUT = "day_out"
    EVENT = "event"
    EVERYDAY = "everyday"
    PERSON = "person"
    THEME = "theme"
    UNSORTED = "unsorted"


@dataclass
class GeoMetadata:
    """EXIF-derived capture metadata for one image.

    Any field may be ``None`` when the source photo lacks that EXIF tag;
    callers must treat missing geo/time as a normal case (FR-011).
    """

    image_path: str
    timestamp: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    @property
    def has_geo(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def has_time(self) -> bool:
        return self.timestamp is not None


@dataclass
class Segment:
    """A candidate group of images produced by one axis."""

    image_paths: list[str]
    kind: Optional[SegmentKind] = None
    label: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.image_paths)


@dataclass
class Segmentation:
    """One axis's proposed grouping of an album.

    ``score_space`` maps each covered image path to a coordinate vector in the
    axis's own space (e.g. radians lat/lon for geo, seconds for time); the
    scorer uses it + ``metric`` to measure how separated the segments are.
    ``metric`` of ``"none"`` means the axis is categorical (e.g. identity) and
    separation is not measured geometrically.
    """

    axis: str
    segments: list[Segment]
    unsorted: list[str] = field(default_factory=list)
    score_space: dict[str, Any] = field(default_factory=dict)
    metric: str = "euclidean"

    def path_to_label(self) -> dict[str, int]:
        """Map each covered image path -> its segment index (for stability/ARI)."""
        out: dict[str, int] = {}
        for i, seg in enumerate(self.segments):
            for p in seg.image_paths:
                out[p] = i
        return out


@dataclass
class AxisScore:
    """The 0–1 quality of one axis's segmentation, with its components."""

    axis: str
    overall: float
    separation: Optional[float] = None
    coverage: float = 0.0
    balance: Optional[float] = None
    stability: Optional[float] = None
    parsimony: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationOutcome:
    """Result of the competition: a winning Segmentation, or FLAT."""

    winner: Optional[Segmentation]
    scores: list[AxisScore]
    flat: bool
    floor: float

    @property
    def winning_axis(self) -> Optional[str]:
        return self.winner.axis if self.winner else None
