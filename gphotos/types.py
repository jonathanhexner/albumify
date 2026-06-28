"""Google Photos Picker API types (spec-092).

Framework-agnostic dataclasses shared by the picker client, the download cache,
and the standalone smoke scripts. No dependency on PipelineContext -- the app/
pipeline is the translator (mirrors the geo_cluster convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _parse_duration(value: Optional[str], default: float) -> float:
    """Parse a Google protobuf Duration (e.g. ``"3.5s"``) into seconds."""
    if not value:
        return default
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return default


@dataclass
class PollingConfig:
    """Server-recommended polling cadence for a picker session."""

    poll_interval_s: float = 3.0
    timeout_s: float = 1800.0


@dataclass
class PickerSession:
    """A Photos Picker session the user completes in their browser."""

    id: str
    picker_uri: str
    media_items_set: bool = False
    polling: PollingConfig = field(default_factory=PollingConfig)
    expire_time: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "PickerSession":
        pc = data.get("pollingConfig") or {}
        return cls(
            id=data["id"],
            picker_uri=data.get("pickerUri", ""),
            media_items_set=bool(data.get("mediaItemsSet", False)),
            polling=PollingConfig(
                poll_interval_s=_parse_duration(pc.get("pollInterval"), 3.0),
                timeout_s=_parse_duration(pc.get("timeoutIn"), 1800.0),
            ),
            expire_time=data.get("expireTime"),
        )


@dataclass
class PickedItem:
    """One photo/video the user picked, with its download base URL."""

    id: str
    filename: str
    base_url: str
    mime_type: str
    media_type: str = "TYPE_UNSPECIFIED"  # PHOTO | VIDEO
    create_time: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict) -> "PickedItem":
        mf = data.get("mediaFile") or {}
        return cls(
            id=data["id"],
            filename=mf.get("filename", data["id"]),
            base_url=mf.get("baseUrl", ""),
            mime_type=mf.get("mimeType", ""),
            media_type=data.get("type", "TYPE_UNSPECIFIED"),
            create_time=data.get("createTime"),
        )

    @property
    def is_video(self) -> bool:
        return self.media_type == "VIDEO"


@dataclass
class DownloadedItem:
    """A picked item after its bytes have been written to the local cache."""

    item: PickedItem
    local_path: str
