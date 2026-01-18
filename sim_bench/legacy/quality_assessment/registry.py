"""
Clean factory registry for quality assessment methods.

Methods register themselves, factory just looks them up.
No messy if/else chains.
"""

from typing import Dict, Type, Any
import logging

from sim_bench.quality_assessment.base import QualityAssessor

logger = logging.getLogger(__name__)


class QualityMethodRegistry:
    """Registry for quality assessment methods."""

    _methods: Dict[str, Type[QualityAssessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a quality assessment method.

        Usage:
            @QualityMethodRegistry.register('rule_based')
            class RuleBasedQuality(QualityAssessor):
                ...
        """
        def decorator(method_class: Type[QualityAssessor]):
            cls._methods[name] = method_class
            return method_class
        return decorator

    @classmethod
    def create(cls, method_type: str, config: Dict[str, Any]) -> QualityAssessor:
        """
        Create quality assessor from config.

        Args:
            method_type: Type of method (e.g., 'rule_based', 'clip_aesthetic')
            config: Configuration dict (method pulls what it needs)

        Returns:
            Configured QualityAssessor instance

        Raises:
            ValueError: If method type not registered or not available
        """
        if method_type not in cls._methods:
            available = ', '.join(cls._methods.keys())
            raise ValueError(
                f"Unknown method type: '{method_type}'. "
                f"Available types: {available}"
            )

        method_class = cls._methods[method_type]

        # Check if dependencies available
        if not method_class.is_available():
            raise ImportError(
                f"Method '{method_type}' dependencies not available. "
                f"Check installation requirements."
            )

        # Create instance using from_config
        return method_class.from_config(config)

    @classmethod
    def list_available(cls) -> Dict[str, bool]:
        """
        List all registered methods and their availability.

        Returns:
            Dict mapping method name -> is_available
        """
        return {
            name: method_class.is_available()
            for name, method_class in cls._methods.items()
        }

    @classmethod
    def get_method_class(cls, name: str) -> Type[QualityAssessor]:
        """Get method class by name."""
        return cls._methods.get(name)


# Convenience alias
register_method = QualityMethodRegistry.register


def create_quality_assessor(method_config: Dict[str, Any]) -> QualityAssessor:
    """
    Convenience function for creating quality assessor from config.

    Args:
        method_config: Config dict with 'type' key and method-specific params

    Returns:
        QualityAssessor instance

    Example:
        >>> config = {'type': 'rule_based', 'weights': {...}}
        >>> assessor = create_quality_assessor(config)
    """
    method_type = method_config.get('type')
    if not method_type:
        raise ValueError("Config must have 'type' key")

    # Remove 'type' and 'name' from config before passing to method
    config_copy = method_config.copy()
    config_copy.pop('type', None)
    config_copy.pop('name', None)

    return QualityMethodRegistry.create(method_type, config_copy)
