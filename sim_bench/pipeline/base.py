"""Pipeline step protocol and metadata definitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING, Dict, List, Any, Optional

if TYPE_CHECKING:
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.cache_handler import CacheKey, UniversalCacheHandler


@dataclass
class StepMetadata:
    """Metadata describing a pipeline step for discovery and validation."""

    name: str
    display_name: str
    description: str
    category: str  # "analysis", "filtering", "embedding", "clustering", "selection"

    requires: set[str] = field(default_factory=set)
    produces: set[str] = field(default_factory=set)
    depends_on: list[str] = field(default_factory=list)

    config_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "category": self.category,
            "requires": list(self.requires),
            "produces": list(self.produces),
            "depends_on": self.depends_on,
            "config_schema": self.config_schema,
        }


class PipelineStep(Protocol):
    """Protocol defining the interface for all pipeline steps."""

    @property
    def metadata(self) -> StepMetadata:
        """Return step metadata for discovery and validation."""
        ...

    def process(self, context: "PipelineContext", config: dict) -> None:
        """
        Execute this step.

        Reads from context what it needs (metadata.requires).
        Writes to context what it produces (metadata.produces).

        Args:
            context: Shared pipeline context with all accumulated data
            config: Step-specific configuration dictionary
        """
        ...

    def validate(self, context: "PipelineContext") -> list[str]:
        """
        Validate that required inputs exist in context.

        Args:
            context: Shared pipeline context

        Returns:
            List of error messages (empty if valid)
        """
        ...


class BaseStep(ABC):
    """
    Base class for pipeline steps with template method pattern for caching.
    
    Steps can either:
    1. Override process() directly (no caching, full control)
    2. Implement abstract methods to use template method (automatic caching)
    """

    _metadata: StepMetadata

    @property
    def metadata(self) -> StepMetadata:
        return self._metadata

    def validate(self, context: "PipelineContext") -> list[str]:
        """Default validation checks that required context keys exist."""
        errors = []
        for key in self._metadata.requires:
            value = getattr(context, key, None)
            if value is None:
                errors.append(f"Missing required context key: {key}")
            elif isinstance(value, (list, dict, set)) and len(value) == 0:
                errors.append(f"Required context key is empty: {key}")
        return errors
    
    def process(self, context: "PipelineContext", config: dict) -> None:
        """
        Template method - handles caching flow automatically.
        
        Steps can override this to handle caching themselves, or implement
        the abstract methods to use the template method pattern.
        """
        # Check if step wants to use template method
        if self._uses_template_method():
            self._process_with_cache(context, config)
        else:
            # Fall back to direct processing (for backward compatibility)
            self._process_direct(context, config)
    
    def _uses_template_method(self) -> bool:
        """
        Check if step implements template method pattern.
        
        Returns True if step implements _get_cache_config, False otherwise.
        """
        # Check if step has implemented cache-related methods
        return (
            hasattr(self, '_get_cache_config') and
            callable(getattr(self, '_get_cache_config', None))
        )
    
    def _process_with_cache(self, context: "PipelineContext", config: dict) -> None:
        """Template method implementation - handles caching flow."""
        from sim_bench.pipeline.cache_handler import UniversalCacheHandler, CacheKey
        
        # Get cache configuration from step
        cache_config = self._get_cache_config(context, config)
        if not cache_config:
            # Step doesn't want caching, process directly
            self._process_uncached_all(context, config)
            return
        
        items = cache_config["items"]
        if not items:
            context.report_progress(self._metadata.name, 1.0, "No items to process")
            return
        
        # Get cache handler
        cache_handler = self._get_cache_handler(context)
        if not cache_handler:
            # No cache available, process all items
            self._process_uncached_all(context, config)
            return
        
        # Build cache keys
        feature_type = cache_config["feature_type"]
        model_name = cache_config["model_name"]
        cache_keys = [
            CacheKey(image_path=item, feature_type=feature_type, model_name=model_name)
            for item in items
        ]
        
        # Load from cache
        cached_data = cache_handler.load_from_cache(cache_keys)
        
        # Find uncached items
        uncached_items = []
        cached_results = {}
        for item, key in zip(items, cache_keys):
            key_str = f"{key.image_path}:{key.feature_type}:{key.model_name}"
            if key_str in cached_data:
                # Deserialize cached data
                data_bytes, metadata = cached_data[key_str]
                result = self._deserialize_from_cache(data_bytes, item)
                cached_results[item] = result
            else:
                uncached_items.append(item)
        
        # Report cache statistics
        cache_hits = len(items) - len(uncached_items)
        if cache_hits > 0:
            context.report_progress(
                self._metadata.name, 0.01,
                f"Cache: {cache_hits} hits, {len(uncached_items)} misses"
            )
        
        # Process uncached items
        if uncached_items:
            new_results = self._process_uncached(uncached_items, context, config)
            
            # Serialize and save to cache
            metadata = cache_config.get("metadata", {})
            for item, result in new_results.items():
                data_bytes = self._serialize_for_cache(result, item)
                key = CacheKey(
                    image_path=item,
                    feature_type=feature_type,
                    model_name=model_name
                )
                cache_handler.store_to_cache(key, data_bytes, metadata)
            
            cached_results.update(new_results)
        
        # Store results in context
        self._store_results(context, cached_results, config)
        
        context.report_progress(
            self._metadata.name, 1.0,
            f"Completed {len(items)} items ({len(uncached_items)} computed, {cache_hits} cached)"
        )
    
    def _process_direct(self, context: "PipelineContext", config: dict) -> None:
        """Fallback for steps that override process() directly."""
        # This should not be called if step uses template method
        # Steps that override process() handle everything themselves
        pass
    
    def _get_cache_handler(self, context: "PipelineContext") -> Optional["UniversalCacheHandler"]:
        """Get cache handler from context."""
        # Check if context has cache_handler (new) or cache_service (old)
        if hasattr(context, 'cache_handler') and context.cache_handler:
            return context.cache_handler
        return None
    
    # ============================================================================
    # Abstract methods for template method pattern
    # ============================================================================
    
    def _get_cache_config(
        self,
        context: "PipelineContext",
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Get cache configuration for this step.
        
        Returns None if step doesn't want caching, or dict with:
            - items: List[str] - items to process
            - feature_type: str - cache feature type
            - model_name: str - model identifier
            - metadata: Optional[Dict] - additional metadata
        
        Override this to enable caching for the step.
        """
        return None
    
    def _process_uncached(
        self,
        items: List[str],
        context: "PipelineContext",
        config: dict
    ) -> Dict[str, Any]:
        """
        Process uncached items.
        
        Args:
            items: List of item keys (e.g., image paths) to process
            context: Pipeline context
            config: Step configuration
        
        Returns:
            Dict mapping item -> result (step-specific type)
        """
        raise NotImplementedError("Step must implement _process_uncached()")
    
    def _process_uncached_all(
        self,
        context: "PipelineContext",
        config: dict
    ) -> None:
        """
        Process all items without caching (fallback when cache unavailable).
        
        Default implementation calls _process_uncached with all items.
        Steps can override for custom behavior.
        """
        cache_config = self._get_cache_config(context, config)
        if cache_config:
            items = cache_config["items"]
            results = self._process_uncached(items, context, config)
            self._store_results(context, results, config)
    
    def _serialize_for_cache(self, result: Any, item: str) -> bytes:
        """
        Serialize result to bytes for caching.
        
        Args:
            result: Step-specific result object
            item: Item key (for context)
        
        Returns:
            Serialized bytes
        """
        raise NotImplementedError("Step must implement _serialize_for_cache()")
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Any:
        """
        Deserialize bytes from cache to result object.
        
        Args:
            data: Cached bytes
            item: Item key (for context)
        
        Returns:
            Deserialized result object
        """
        raise NotImplementedError("Step must implement _deserialize_from_cache()")
    
    def _store_results(
        self,
        context: "PipelineContext",
        results: Dict[str, Any],
        config: dict
    ) -> None:
        """
        Store results in pipeline context.
        
        Args:
            context: Pipeline context
            results: Dict mapping items to results
            config: Step configuration
        """
        raise NotImplementedError("Step must implement _store_results()")