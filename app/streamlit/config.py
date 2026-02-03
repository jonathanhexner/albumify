"""Streamlit app configuration."""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AppConfig:
    """Configuration for the Streamlit frontend app."""

    # API settings
    api_base_url: str = "http://localhost:8000"
    api_timeout_sec: int = 30

    # Polling settings
    poll_interval_sec: float = 1.0
    max_poll_attempts: int = 600  # 10 minutes max

    # UI settings
    page_title: str = "Album Organizer"
    page_icon: str = ":camera:"
    layout: str = "wide"

    # Gallery settings
    images_per_row: int = 4
    thumbnail_size: tuple = (200, 200)

    # People browser settings
    people_images_per_row: int = 5
    min_face_confidence: float = 0.7

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        return cls(
            api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
            api_timeout_sec=int(os.getenv("API_TIMEOUT_SEC", "30")),
            poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "1.0")),
            page_title=os.getenv("PAGE_TITLE", "Album Organizer"),
            images_per_row=int(os.getenv("IMAGES_PER_ROW", "4")),
        )


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global config instance."""
    global _config
    _config = config
