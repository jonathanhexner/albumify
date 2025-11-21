"""
Tool registry using Registry pattern.

Auto-discovers and manages all available tools for agent orchestration.
"""

from typing import Dict, List, Type, Optional
from pathlib import Path
import importlib
import inspect
import logging

from sim_bench.agent.tools.base import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for discovering and loading agent tools.

    Features:
    - Auto-discovery from *_tools.py files
    - Singleton instances per configuration
    - Tool schema retrieval for LLM
    - Category-based filtering
    """

    def __init__(self):
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
        self._discovered = False

    def discover_tools(self, tools_dir: Optional[Path] = None):
        """
        Auto-discover tools from *_tools.py files.

        Args:
            tools_dir: Directory to search (defaults to this module's directory)
        """
        if self._discovered:
            logger.info("Tools already discovered, skipping")
            return

        if tools_dir is None:
            tools_dir = Path(__file__).parent

        logger.info(f"Discovering tools in {tools_dir}")

        # Find all *_tools.py files
        tool_files = list(tools_dir.glob("*_tools.py"))
        logger.info(f"Found {len(tool_files)} tool modules")

        for tool_file in tool_files:
            self._import_tool_module(tool_file)

        self._discovered = True
        logger.info(f"Discovered {len(self._tool_classes)} tools total")

    def _import_tool_module(self, tool_file: Path):
        """Import a tool module and register all tool classes."""
        module_name = f"sim_bench.agent.tools.{tool_file.stem}"

        try:
            module = importlib.import_module(module_name)

            # Find all BaseTool subclasses in module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTool) and
                    obj is not BaseTool and
                    obj.__module__ == module_name):

                    tool_name = obj.get_name()
                    self._tool_classes[tool_name] = obj
                    logger.info(f"Registered tool: {tool_name} from {module_name}")

        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}", exc_info=True)

    def register(self, tool_class: Type[BaseTool]):
        """
        Manually register a tool class.

        Args:
            tool_class: Tool class to register
        """
        tool_name = tool_class.get_name()
        self._tool_classes[tool_name] = tool_class
        logger.info(f"Manually registered tool: {tool_name}")

    def get_tool(self, name: str, config: Dict = None) -> BaseTool:
        """
        Get tool instance (singleton per config).

        Args:
            name: Tool name
            config: Tool configuration

        Returns:
            Tool instance

        Raises:
            ValueError: If tool not found
        """
        if name not in self._tool_classes:
            raise ValueError(
                f"Unknown tool: {name}. "
                f"Available tools: {', '.join(self.list_tools())}"
            )

        # Create cache key from config
        config = config or {}
        cache_key = f"{name}_{hash(frozenset(config.items()))}"

        # Return cached instance or create new
        if cache_key not in self._tool_instances:
            tool_class = self._tool_classes[name]
            self._tool_instances[cache_key] = tool_class(config)
            logger.debug(f"Created new instance of {name}")

        return self._tool_instances[cache_key]

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        List all registered tool names.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        if category is None:
            return sorted(self._tool_classes.keys())

        # Filter by category
        filtered = []
        for name, tool_class in self._tool_classes.items():
            if tool_class.get_category() == category:
                filtered.append(name)

        return sorted(filtered)

    def get_tools_by_category(self) -> Dict[ToolCategory, List[str]]:
        """
        Group tools by category.

        Returns:
            Dictionary mapping category to list of tool names
        """
        categorized = {}

        for name, tool_class in self._tool_classes.items():
            category = tool_class.get_category()
            categorized.setdefault(category, []).append(name)

        # Sort each category's tools
        for category in categorized:
            categorized[category] = sorted(categorized[category])

        return categorized

    def get_tool_schema(self, name: str) -> Dict:
        """
        Get tool schema for LLM function calling.

        Args:
            name: Tool name

        Returns:
            OpenAI function calling compatible schema

        Raises:
            ValueError: If tool not found
        """
        if name not in self._tool_classes:
            raise ValueError(f"Unknown tool: {name}")

        return self._tool_classes[name].get_schema()

    def get_all_schemas(self, category: Optional[ToolCategory] = None) -> List[Dict]:
        """
        Get schemas for all tools (or filtered by category).

        Args:
            category: Optional category filter

        Returns:
            List of tool schemas
        """
        tools = self.list_tools(category)
        return [self.get_tool_schema(name) for name in tools]

    def get_tool_examples(self, name: str) -> List[Dict]:
        """
        Get usage examples for a tool.

        Args:
            name: Tool name

        Returns:
            List of example dictionaries

        Raises:
            ValueError: If tool not found
        """
        if name not in self._tool_classes:
            raise ValueError(f"Unknown tool: {name}")

        return self._tool_classes[name].get_examples()

    def get_tool_info(self, name: str) -> Dict:
        """
        Get comprehensive tool information.

        Args:
            name: Tool name

        Returns:
            Dictionary with schema, examples, category, etc.

        Raises:
            ValueError: If tool not found
        """
        if name not in self._tool_classes:
            raise ValueError(f"Unknown tool: {name}")

        tool_class = self._tool_classes[name]

        return {
            'name': name,
            'description': tool_class.get_description(),
            'category': tool_class.get_category().value,
            'schema': tool_class.get_schema(),
            'examples': tool_class.get_examples()
        }

    def clear_instances(self):
        """Clear all cached tool instances."""
        self._tool_instances.clear()
        logger.info("Cleared all tool instances")

    def get_statistics(self) -> Dict:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        categorized = self.get_tools_by_category()

        return {
            'total_tools': len(self._tool_classes),
            'cached_instances': len(self._tool_instances),
            'tools_by_category': {
                category.value: len(tools)
                for category, tools in categorized.items()
            },
            'discovered': self._discovered
        }


# Global registry instance (singleton)
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get global tool registry instance (singleton).

    Returns:
        Global ToolRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = ToolRegistry()
        _global_registry.discover_tools()

    return _global_registry
