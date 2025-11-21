"""
Abstract base class for agent tools using Template Method pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and discovery."""
    SIMILARITY = "similarity"
    CLUSTERING = "clustering"
    QUALITY = "quality"
    ANALYSIS = "analysis"
    ORGANIZATION = "organization"
    LAYOUT = "layout"
    EXPORT = "export"


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Tools wrap sim-bench capabilities with:
    - Standardized execute() interface
    - Schema for LLM function calling
    - Parameter validation
    - Error handling and logging

    Uses Template Method pattern - subclasses implement execute()
    while base class handles validation and error handling.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup()

    def setup(self):
        """
        Override to perform tool-specific initialization.

        Called during __init__ after config is set.
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Template method that must be implemented by subclasses.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Dictionary with standardized format:
            {
                'success': bool,           # Whether execution succeeded
                'data': Any,               # Tool-specific result data
                'message': str,            # Human-readable description
                'metadata': Dict           # Additional info (timing, etc.)
            }
        """
        pass

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Dict:
        """
        Get tool schema for LLM function calling.

        Must follow OpenAI function calling schema format.

        Returns:
            Schema dictionary:
            {
                'name': str,                    # Tool name (snake_case)
                'description': str,             # What the tool does
                'category': ToolCategory,       # Tool category
                'parameters': {                 # Parameters schema
                    'type': 'object',
                    'properties': {
                        'param_name': {
                            'type': 'string',
                            'description': '...',
                            'enum': [...],      # Optional
                            'default': ...      # Optional
                        }
                    },
                    'required': [...]           # Required parameters
                }
            }
        """
        pass

    @classmethod
    @abstractmethod
    def get_examples(cls) -> List[Dict]:
        """
        Get example usage scenarios for this tool.

        Used for:
        - LLM few-shot prompting
        - Documentation
        - Testing

        Returns:
            List of example dictionaries:
            [
                {
                    'query': 'User natural language query',
                    'params': {'param1': 'value1', ...},
                    'description': 'What this example demonstrates'
                },
                ...
            ]
        """
        pass

    def validate_params(self, **kwargs) -> bool:
        """
        Validate parameters against tool schema.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        schema = self.get_schema()
        required = schema['parameters'].get('required', [])

        # Check required parameters
        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        # Check parameter types (basic validation)
        properties = schema['parameters'].get('properties', {})
        for param, value in kwargs.items():
            if param in properties:
                expected_type = properties[param].get('type')
                if not self._validate_type(value, expected_type):
                    raise ValueError(
                        f"Parameter '{param}' has incorrect type. "
                        f"Expected {expected_type}, got {type(value).__name__}"
                    )

        return True

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value matches expected JSON schema type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if not expected_python_type:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_python_type)

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool with validation and error handling (Template Method).

        This is the main entry point that:
        1. Validates parameters
        2. Calls execute()
        3. Handles errors
        4. Returns standardized result

        Args:
            **kwargs: Tool parameters

        Returns:
            Standardized result dictionary
        """
        try:
            # Validate parameters
            self.validate_params(**kwargs)

            # Execute tool
            self.logger.info(f"Executing {self.__class__.__name__}")
            result = self.execute(**kwargs)

            # Ensure result has required fields
            if not isinstance(result, dict):
                raise ValueError("Tool must return dictionary")

            if 'success' not in result:
                result['success'] = True

            if 'message' not in result:
                result['message'] = f"{self.__class__.__name__} completed"

            if 'metadata' not in result:
                result['metadata'] = {}

            self.logger.info(f"{self.__class__.__name__} completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"{self.__class__.__name__} failed: {e}", exc_info=True)
            return {
                'success': False,
                'data': None,
                'message': f"Tool execution failed: {str(e)}",
                'metadata': {'error': str(e), 'error_type': type(e).__name__}
            }

    @classmethod
    def get_name(cls) -> str:
        """Get tool name from schema."""
        return cls.get_schema()['name']

    @classmethod
    def get_category(cls) -> ToolCategory:
        """Get tool category from schema."""
        return cls.get_schema()['category']

    @classmethod
    def get_description(cls) -> str:
        """Get tool description from schema."""
        return cls.get_schema()['description']
