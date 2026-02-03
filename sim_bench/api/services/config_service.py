"""Config service - manages configuration profiles."""

import logging
import uuid
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from sim_bench.api.database.models import ConfigProfile

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "pipeline.yaml"
DEFAULT_PROFILE_NAME = "default"


def load_yaml_config(path: Path) -> dict:
    """Load configuration from YAML file."""
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_default_config() -> dict:
    """Get the default configuration from YAML file."""
    return load_yaml_config(DEFAULT_CONFIG_PATH)


class ConfigService:
    """Service for managing configuration profiles."""

    def __init__(self, session: Session):
        self._session = session
        self._logger = logging.getLogger(__name__)

    def ensure_default_profile(self) -> ConfigProfile:
        """Ensure the default profile exists, creating it from YAML if needed."""
        profile = self._session.query(ConfigProfile).filter(
            ConfigProfile.name == DEFAULT_PROFILE_NAME
        ).first()

        if profile is None:
            self._logger.info("Creating default config profile from pipeline.yaml")
            default_config = get_default_config()
            profile = ConfigProfile(
                id=str(uuid.uuid4()),
                name=DEFAULT_PROFILE_NAME,
                description="Default configuration loaded from pipeline.yaml",
                config=default_config,
                is_default=True
            )
            self._session.add(profile)
            self._session.commit()
            self._logger.info(f"Created default profile: {profile.id}")

        return profile

    def list_profiles(self) -> list[ConfigProfile]:
        """List all configuration profiles."""
        return self._session.query(ConfigProfile).order_by(ConfigProfile.name).all()

    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """Get a profile by name."""
        return self._session.query(ConfigProfile).filter(
            ConfigProfile.name == name
        ).first()

    def get_profile_by_id(self, profile_id: str) -> Optional[ConfigProfile]:
        """Get a profile by ID."""
        return self._session.query(ConfigProfile).filter(
            ConfigProfile.id == profile_id
        ).first()

    def get_default_profile(self) -> Optional[ConfigProfile]:
        """Get the default profile."""
        profile = self._session.query(ConfigProfile).filter(
            ConfigProfile.is_default == True
        ).first()

        if profile is None:
            profile = self.ensure_default_profile()

        return profile

    def create_profile(
        self,
        name: str,
        config: dict,
        description: str = None,
        base_profile: str = None
    ) -> ConfigProfile:
        """Create a new configuration profile."""
        existing = self.get_profile(name)
        if existing:
            raise ValueError(f"Profile '{name}' already exists")

        # Start with base profile config if specified
        if base_profile:
            base = self.get_profile(base_profile)
            if base:
                merged_config = self._deep_merge(base.config, config)
            else:
                merged_config = config
        else:
            # Start with default config and merge overrides
            default_config = get_default_config()
            merged_config = self._deep_merge(default_config, config)

        profile = ConfigProfile(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            config=merged_config,
            is_default=False
        )
        self._session.add(profile)
        self._session.commit()
        self._logger.info(f"Created profile '{name}': {profile.id}")
        return profile

    def update_profile(
        self,
        name: str,
        config: dict = None,
        description: str = None
    ) -> ConfigProfile:
        """Update an existing profile."""
        profile = self.get_profile(name)
        if profile is None:
            raise ValueError(f"Profile '{name}' not found")

        if config is not None:
            profile.config = self._deep_merge(profile.config, config)
        if description is not None:
            profile.description = description

        self._session.commit()
        self._logger.info(f"Updated profile '{name}'")
        return profile

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""
        if name == DEFAULT_PROFILE_NAME:
            raise ValueError("Cannot delete the default profile")

        profile = self.get_profile(name)
        if profile is None:
            return False

        self._session.delete(profile)
        self._session.commit()
        self._logger.info(f"Deleted profile '{name}'")
        return True

    def reset_profile_to_defaults(self, name: str) -> ConfigProfile:
        """Reset a profile to default values from YAML."""
        profile = self.get_profile(name)
        if profile is None:
            raise ValueError(f"Profile '{name}' not found")

        profile.config = get_default_config()
        self._session.commit()
        self._logger.info(f"Reset profile '{name}' to defaults")
        return profile

    def duplicate_profile(self, source_name: str, new_name: str) -> ConfigProfile:
        """Duplicate a profile with a new name."""
        source = self.get_profile(source_name)
        if source is None:
            raise ValueError(f"Profile '{source_name}' not found")

        return self.create_profile(
            name=new_name,
            config=source.config.copy(),
            description=f"Copy of {source_name}"
        )

    def get_merged_config(self, profile_name: str = None, overrides: dict = None) -> dict:
        """Get merged config: profile + runtime overrides."""
        if profile_name:
            profile = self.get_profile(profile_name)
        else:
            profile = self.get_default_profile()

        if profile is None:
            base_config = get_default_config()
        else:
            base_config = profile.config.copy()

        if overrides:
            return self._deep_merge(base_config, overrides)
        return base_config

    def _deep_merge(self, base: dict, overrides: dict) -> dict:
        """Deep merge two dictionaries, with overrides taking precedence."""
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
