"""
Global configuration management for sim-bench.

Implements Singleton pattern to ensure single source of truth for global settings.
Configuration hierarchy (highest to lowest priority):
1. CLI arguments (passed directly to functions)
2. Experiment-specific config files
3. Global config file (configs/global_config.yaml)
4. Hardcoded defaults

Example:
    >>> from sim_bench.config import get_global_config
    >>> config = get_global_config()
    >>> device = config.get('device')
    >>> output_dir = config.get_path('output_dir')
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


logger = logging.getLogger(__name__)


class GlobalConfig:
    """
    Singleton class for global configuration management.

    Loads configuration from configs/global_config.yaml and provides
    thread-safe access to settings across all modules.
    """

    _instance: Optional['GlobalConfig'] = None
    _config: Dict[str, Any] = {}
    _loaded: bool = False

    def __new__(cls) -> 'GlobalConfig':
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration (only loads once)."""
        if not self._loaded:
            self._load_config()
            self._loaded = True

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        # Find project root (where configs/ directory is located)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent  # sim-bench/sim_bench/config.py -> sim-bench/

        config_path = project_root / 'configs' / 'global_config.yaml'

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"Loaded global config from: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load global config: {e}. Using defaults.")
                self._config = self._get_default_config()
        else:
            logger.warning(f"Global config not found at {config_path}. Using defaults.")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return hardcoded default configuration."""
        return {
            'device': 'cpu',
            'num_workers': 4,
            'output_dir': 'outputs/',
            'cache_dir': '.cache/',
            'log_dir': 'logs/',
            'enable_embedding_cache': True,
            'enable_thumbnail_cache': True,
            'enable_quality_cache': True,
            'thumbnail_sizes': {
                'tiny': 128,
                'small': 512,
                'medium': 1024,
                'large': 2048
            },
            'clip': {
                'model_name': 'ViT-B-32',
                'pretrained': 'laion2b_s34b_b79k',
                'batch_size': 32
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_to_console': True
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dots).

        Args:
            key: Configuration key (e.g., 'device' or 'clip.model_name')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = get_global_config()
            >>> device = config.get('device', 'cpu')
            >>> clip_model = config.get('clip.model_name', 'ViT-B-32')
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str, default: str = '') -> Path:
        """
        Get configuration value as Path object.

        Args:
            key: Configuration key for a path
            default: Default path if key not found

        Returns:
            Path object

        Example:
            >>> config = get_global_config()
            >>> output_dir = config.get_path('output_dir')
        """
        path_str = self.get(key, default)
        return Path(path_str)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value (runtime only, not persisted).

        Args:
            key: Configuration key (supports nested with dots)
            value: Value to set

        Example:
            >>> config = get_global_config()
            >>> config.set('device', 'cuda')
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def reload(self) -> None:
        """Reload configuration from file."""
        self._loaded = False
        self._load_config()
        self._loaded = True
        logger.info("Global configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()

    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration dict into global config.

        Args:
            other_config: Dictionary to merge (overwrites existing values)

        Example:
            >>> config = get_global_config()
            >>> config.merge({'device': 'cuda', 'num_workers': 8})
        """
        self._deep_merge(self._config, other_config)

    def _deep_merge(self, base: Dict, updates: Dict) -> None:
        """Recursively merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# Singleton instance accessor
_global_config_instance: Optional[GlobalConfig] = None


def get_global_config() -> GlobalConfig:
    """
    Get the global configuration instance (Singleton).

    Returns:
        GlobalConfig instance

    Example:
        >>> from sim_bench.config import get_global_config
        >>> config = get_global_config()
        >>> device = config.get('device')
    """
    global _global_config_instance
    if _global_config_instance is None:
        _global_config_instance = GlobalConfig()
    return _global_config_instance


def setup_logging() -> None:
    """
    Setup logging based on global configuration.

    Should be called once at application startup.

    Example:
        >>> from sim_bench.config import setup_logging
        >>> setup_logging()
    """
    config = get_global_config()

    # Get logging settings
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get(
        'logging.format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    date_format = config.get('logging.date_format', '%Y-%m-%d %H:%M:%S')
    log_to_file = config.get_bool('logging.log_to_file', True)
    log_to_console = config.get_bool('logging.log_to_console', True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler
    if log_to_file:
        log_dir = config.get_path('log_dir')
        log_dir.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        max_bytes = config.get_int('logging.max_log_size_mb', 100) * 1024 * 1024
        backup_count = config.get_int('logging.backup_count', 3)

        file_handler = RotatingFileHandler(
            log_dir / 'sim-bench.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Logging configured: level={log_level}, console={log_to_console}, file={log_to_file}")


def merge_configs(*configs: Union[Dict, Path, str]) -> Dict[str, Any]:
    """
    Merge multiple configurations with proper precedence.

    Args:
        *configs: Config dicts, YAML file paths, or config names
            Later configs override earlier ones

    Returns:
        Merged configuration dictionary

    Example:
        >>> from sim_bench.config import merge_configs, get_global_config
        >>> global_cfg = get_global_config().to_dict()
        >>> experiment_cfg = {'device': 'cuda', 'batch_size': 64}
        >>> final_cfg = merge_configs(global_cfg, experiment_cfg)
        >>> # final_cfg has device='cuda' (overridden), other global values preserved
    """
    merged = {}

    for cfg in configs:
        if isinstance(cfg, (Path, str)):
            # Load from YAML file
            cfg_path = Path(cfg)
            if not cfg_path.is_absolute():
                # Relative to configs/ directory
                project_root = Path(__file__).parent.parent
                cfg_path = project_root / 'configs' / cfg_path

            with open(cfg_path, 'r') as f:
                cfg_dict = yaml.safe_load(f) or {}
        elif isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            raise TypeError(f"Config must be dict or Path, got {type(cfg)}")

        # Deep merge
        _deep_merge_dicts(merged, cfg_dict)

    return merged


def _deep_merge_dicts(base: Dict, updates: Dict) -> None:
    """Helper for deep dictionary merge (in-place)."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, returning a new dictionary.

    Args:
        base: Base configuration dictionary
        updates: Updates to apply (takes precedence)

    Returns:
        New dictionary with base values updated by updates

    Example:
        >>> base = {'a': 1, 'nested': {'b': 2, 'c': 3}}
        >>> updates = {'nested': {'b': 20, 'd': 4}}
        >>> result = deep_merge(base, updates)
        >>> # result = {'a': 1, 'nested': {'b': 20, 'c': 3, 'd': 4}}
    """
    import copy
    result = copy.deepcopy(base)
    _deep_merge_dicts(result, updates)
    return result


__all__ = [
    'GlobalConfig',
    'get_global_config',
    'setup_logging',
    'merge_configs',
    'deep_merge'
]
