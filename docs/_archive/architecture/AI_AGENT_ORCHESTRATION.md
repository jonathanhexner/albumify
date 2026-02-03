# AI Agent Orchestration Architecture

## Overview

Design for AI agent-driven photo organization system where an LLM orchestrates all sim-bench capabilities through natural language interaction.

## Design Principles

1. **Factory Pattern**: All components loadable via factories
2. **Strategy Pattern**: Pluggable execution strategies
3. **Observer Pattern**: State change notifications
4. **Command Pattern**: Tool execution as commands
5. **Chain of Responsibility**: Workflow step delegation
6. **Empty `__init__.py`**: Only docstrings + imports
7. **Minimal Control Flow**: Use patterns over if statements
8. **Logging**: Comprehensive logging, no prints in core code

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  (Streamlit Chat UI, CLI, Web API)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Agent Orchestration Layer                  │
│  - AgentPlanner: Query → Workflow Plan                      │
│  - AgentExecutor: Execute workflow steps                    │
│  - AgentMemory: Conversation state + results                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Tool Registry Layer                     │
│  - ToolRegistry: Discover and load tools                    │
│  - Tool wrappers for all sim-bench capabilities             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Capabilities Layer                   │
│  (Existing: Quality, Clustering, Similarity, Analysis)       │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
sim_bench/agent/
├── __init__.py                      # EMPTY (docstring only)
├── factory.py                       # Agent component factories
│
├── core/
│   ├── __init__.py                  # EMPTY
│   ├── base.py                      # Abstract base classes
│   ├── planner.py                   # AgentPlanner implementation
│   ├── executor.py                  # AgentExecutor implementation
│   └── memory.py                    # AgentMemory (state + history)
│
├── tools/
│   ├── __init__.py                  # EMPTY
│   ├── base.py                      # BaseTool abstract class
│   ├── registry.py                  # ToolRegistry (discover + load)
│   ├── similarity_tools.py          # Image retrieval tools
│   ├── clustering_tools.py          # Clustering tools
│   ├── quality_tools.py             # Quality assessment tools
│   ├── analysis_tools.py            # Photo analysis tools
│   └── organization_tools.py        # File organization tools
│
├── workflows/
│   ├── __init__.py                  # EMPTY
│   ├── base.py                      # WorkflowStep, Workflow classes
│   ├── templates.py                 # Pre-defined workflow templates
│   └── builder.py                   # WorkflowBuilder (from LLM plan)
│
└── llm/
    ├── __init__.py                  # EMPTY
    ├── base.py                      # LLMProvider abstract class
    ├── openai.py                    # OpenAI provider
    ├── anthropic.py                 # Anthropic provider
    └── prompts.py                   # Prompt templates
```

## Core Components

### 1. AgentPlanner (Strategy Pattern)

Converts user query → executable workflow plan.

```python
# sim_bench/agent/core/planner.py

from abc import ABC, abstractmethod
from typing import Dict, List
from sim_bench.agent.workflows.base import Workflow

class PlanningStrategy(ABC):
    """Abstract strategy for workflow planning."""

    @abstractmethod
    def create_plan(self, query: str, context: Dict) -> Workflow:
        """Create workflow from user query."""
        pass

class LLMPlanningStrategy(PlanningStrategy):
    """Uses LLM to generate workflow plan."""

    def create_plan(self, query: str, context: Dict) -> Workflow:
        # LLM generates structured plan
        # Returns Workflow with steps
        pass

class TemplatePlanningStrategy(PlanningStrategy):
    """Uses pre-defined templates."""

    def create_plan(self, query: str, context: Dict) -> Workflow:
        # Match query to template
        # Return instantiated workflow
        pass

class AgentPlanner:
    """Plans workflows using pluggable strategies."""

    def __init__(self, strategy: PlanningStrategy):
        self.strategy = strategy

    def plan(self, query: str, context: Dict) -> Workflow:
        return self.strategy.create_plan(query, context)
```

### 2. AgentExecutor (Command Pattern)

Executes workflow steps as commands.

```python
# sim_bench/agent/core/executor.py

from abc import ABC, abstractmethod
from typing import Any, Dict
from sim_bench.agent.workflows.base import Workflow, WorkflowStep

class ExecutionCommand(ABC):
    """Abstract command for executing a step."""

    @abstractmethod
    def execute(self, input_data: Dict) -> Dict:
        """Execute command and return results."""
        pass

    @abstractmethod
    def undo(self) -> None:
        """Undo command execution."""
        pass

class ToolExecutionCommand(ExecutionCommand):
    """Executes a tool."""

    def __init__(self, tool_name: str, tool_params: Dict):
        self.tool_name = tool_name
        self.tool_params = tool_params
        self.result = None

    def execute(self, input_data: Dict) -> Dict:
        # Load tool from registry
        # Execute with params
        # Store result
        pass

class AgentExecutor:
    """Executes workflows step by step."""

    def __init__(self, tool_registry, memory):
        self.tool_registry = tool_registry
        self.memory = memory
        self.command_history = []

    def execute_workflow(self, workflow: Workflow) -> Dict:
        """Execute all workflow steps."""
        results = {}

        for step in workflow.steps:
            command = self._create_command(step)
            step_result = command.execute(results)

            results[step.name] = step_result
            self.command_history.append(command)

            # Update memory
            self.memory.add_step_result(step.name, step_result)

        return results

    def _create_command(self, step: WorkflowStep) -> ExecutionCommand:
        """Factory method for creating commands."""
        return ToolExecutionCommand(step.tool_name, step.params)
```

### 3. ToolRegistry (Registry Pattern)

Discovers and loads all available tools.

```python
# sim_bench/agent/tools/registry.py

from typing import Dict, List, Type
from pathlib import Path
import logging
from sim_bench.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Registry for discovering and loading tools."""

    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}

    def register(self, name: str, tool_class: Type[BaseTool]):
        """Register a tool class."""
        self._tools[name] = tool_class
        logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str, config: Dict = None) -> BaseTool:
        """Get tool instance (singleton per config)."""
        cache_key = f"{name}_{hash(frozenset(config.items()) if config else 0)}"

        if cache_key not in self._instances:
            tool_class = self._tools.get(name)
            if not tool_class:
                raise ValueError(f"Unknown tool: {name}")

            self._instances[cache_key] = tool_class(config or {})
            logger.info(f"Instantiated tool: {name}")

        return self._instances[cache_key]

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_schema(self, name: str) -> Dict:
        """Get tool schema for LLM function calling."""
        tool_class = self._tools.get(name)
        if not tool_class:
            raise ValueError(f"Unknown tool: {name}")

        return tool_class.get_schema()

    def discover_tools(self):
        """Auto-discover tools from tools/ directory."""
        tools_dir = Path(__file__).parent

        # Import all tool modules
        for tool_file in tools_dir.glob("*_tools.py"):
            module_name = f"sim_bench.agent.tools.{tool_file.stem}"
            __import__(module_name)

        logger.info(f"Discovered {len(self._tools)} tools")
```

### 4. BaseTool (Template Method Pattern)

Abstract base class for all tools.

```python
# sim_bench/agent/tools/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from enum import Enum

class ToolCategory(Enum):
    """Tool categories for organization."""
    SIMILARITY = "similarity"
    CLUSTERING = "clustering"
    QUALITY = "quality"
    ANALYSIS = "analysis"
    ORGANIZATION = "organization"
    LAYOUT = "layout"
    EXPORT = "export"

class BaseTool(ABC):
    """
    Abstract base class for agent tools.

    Tools are wrappers around sim-bench capabilities that provide:
    - Standardized interface for agent execution
    - Schema for LLM function calling
    - Validation and error handling
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.setup()

    def setup(self):
        """Override to perform tool-specific setup."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Returns:
            Dictionary with:
            - 'success': bool
            - 'data': tool-specific results
            - 'message': human-readable description
            - 'metadata': additional info
        """
        pass

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Dict:
        """
        Get tool schema for LLM function calling.

        Returns:
            OpenAI function calling compatible schema:
            {
                'name': str,
                'description': str,
                'category': ToolCategory,
                'parameters': {
                    'type': 'object',
                    'properties': {...},
                    'required': [...]
                }
            }
        """
        pass

    @classmethod
    @abstractmethod
    def get_examples(cls) -> List[Dict]:
        """
        Get example usage scenarios.

        Returns:
            List of examples with:
            - 'query': User natural language query
            - 'params': Tool parameters
            - 'description': What the tool does
        """
        pass

    def validate_params(self, **kwargs) -> bool:
        """Validate parameters against schema."""
        schema = self.get_schema()
        required = schema['parameters'].get('required', [])

        for param in required:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter: {param}")

        return True
```

### 5. Workflow System (Composite Pattern)

```python
# sim_bench/agent/workflows/base.py

from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class WorkflowStep:
    """Single step in a workflow."""
    name: str
    tool_name: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Dict = field(default_factory=dict)

    def is_ready(self, completed_steps: set) -> bool:
        """Check if all dependencies are completed."""
        return all(dep in completed_steps for dep in self.dependencies)

@dataclass
class Workflow:
    """
    Complete workflow with multiple steps.

    Uses Composite pattern - workflow can contain sub-workflows.
    """
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)

    def get_next_step(self, completed_steps: set) -> WorkflowStep:
        """Get next executable step based on dependencies."""
        for step in self.steps:
            if step.status == WorkflowStatus.PENDING and step.is_ready(completed_steps):
                return step
        return None

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.status == WorkflowStatus.COMPLETED for step in self.steps)
```

### 6. AgentMemory (Memento Pattern)

```python
# sim_bench/agent/core/memory.py

from typing import Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class ConversationTurn:
    """Single conversation turn."""
    timestamp: datetime
    user_message: str
    agent_response: str
    workflow_executed: str = None
    results: Dict = field(default_factory=dict)

class AgentMemory:
    """
    Manages conversation history and workflow results.

    Uses Memento pattern to save/restore state.
    """

    def __init__(self):
        self.conversation_history: List[ConversationTurn] = []
        self.workflow_results: Dict[str, Any] = {}
        self.current_context: Dict[str, Any] = {}

    def add_turn(self, user_message: str, agent_response: str,
                 workflow: str = None, results: Dict = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            workflow_executed=workflow,
            results=results or {}
        )
        self.conversation_history.append(turn)

    def add_step_result(self, step_name: str, result: Dict):
        """Store step execution result."""
        self.workflow_results[step_name] = result
        self.current_context[step_name] = result

    def get_context(self) -> Dict:
        """Get current context for LLM."""
        return {
            'recent_results': self.workflow_results,
            'conversation_summary': self._summarize_conversation(),
            'available_data': list(self.workflow_results.keys())
        }

    def _summarize_conversation(self) -> str:
        """Summarize recent conversation."""
        if not self.conversation_history:
            return "No prior conversation"

        recent = self.conversation_history[-3:]
        summary = []
        for turn in recent:
            summary.append(f"User: {turn.user_message[:50]}...")
            summary.append(f"Agent: {turn.agent_response[:50]}...")

        return "\n".join(summary)

    def save_state(self) -> Dict:
        """Save memory state (Memento pattern)."""
        return {
            'conversation_history': self.conversation_history,
            'workflow_results': self.workflow_results,
            'current_context': self.current_context
        }

    def restore_state(self, state: Dict):
        """Restore memory from saved state."""
        self.conversation_history = state.get('conversation_history', [])
        self.workflow_results = state.get('workflow_results', {})
        self.current_context = state.get('current_context', {})
```

## Tool Definitions

### Example: Clustering Tool

```python
# sim_bench/agent/tools/clustering_tools.py

from sim_bench.agent.tools.base import BaseTool, ToolCategory
from sim_bench.clustering.factory import create_clustering_method
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ClusterImagesTool(BaseTool):
    """Tool for clustering images by visual similarity."""

    def setup(self):
        """Initialize clustering method."""
        self.method = None  # Lazy initialization

    def execute(self, image_paths: List[str], method: str = 'dbscan',
                feature_type: str = 'dinov2', **kwargs) -> Dict[str, Any]:
        """
        Cluster images into groups.

        Args:
            image_paths: List of image file paths
            method: Clustering algorithm (dbscan, hdbscan, kmeans)
            feature_type: Feature extractor (dinov2, openclip, histogram)
            **kwargs: Method-specific parameters

        Returns:
            {
                'success': bool,
                'data': {
                    'clusters': Dict[int, List[str]],  # cluster_id -> image_paths
                    'num_clusters': int,
                    'noise_images': List[str]  # For DBSCAN/HDBSCAN
                },
                'message': str,
                'metadata': {...}
            }
        """
        self.validate_params(image_paths=image_paths, method=method)

        logger.info(f"Clustering {len(image_paths)} images with {method}")

        # Create clustering method
        self.method = create_clustering_method(
            method_type=method,
            feature_type=feature_type,
            **kwargs
        )

        # Execute clustering
        labels = self.method.cluster(image_paths)

        # Organize results
        clusters = {}
        noise_images = []

        for img_path, label in zip(image_paths, labels):
            if label == -1:  # Noise
                noise_images.append(img_path)
            else:
                clusters.setdefault(label, []).append(img_path)

        num_clusters = len(clusters)

        return {
            'success': True,
            'data': {
                'clusters': clusters,
                'num_clusters': num_clusters,
                'noise_images': noise_images
            },
            'message': f"Found {num_clusters} clusters from {len(image_paths)} images",
            'metadata': {
                'method': method,
                'feature_type': feature_type,
                'params': kwargs
            }
        }

    @classmethod
    def get_schema(cls) -> Dict:
        """Get tool schema for LLM."""
        return {
            'name': 'cluster_images',
            'description': 'Cluster images into groups based on visual similarity',
            'category': ToolCategory.CLUSTERING,
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_paths': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'List of image file paths to cluster'
                    },
                    'method': {
                        'type': 'string',
                        'enum': ['dbscan', 'hdbscan', 'kmeans'],
                        'description': 'Clustering algorithm to use',
                        'default': 'dbscan'
                    },
                    'feature_type': {
                        'type': 'string',
                        'enum': ['dinov2', 'openclip', 'histogram'],
                        'description': 'Feature extraction method',
                        'default': 'dinov2'
                    },
                    'min_cluster_size': {
                        'type': 'integer',
                        'description': 'Minimum images per cluster (DBSCAN/HDBSCAN)',
                        'default': 5
                    }
                },
                'required': ['image_paths']
            }
        }

    @classmethod
    def get_examples(cls) -> List[Dict]:
        """Get usage examples."""
        return [
            {
                'query': 'Group my vacation photos by event',
                'params': {
                    'image_paths': ['photo1.jpg', 'photo2.jpg', '...'],
                    'method': 'dbscan',
                    'feature_type': 'dinov2',
                    'min_cluster_size': 5
                },
                'description': 'Clusters vacation photos into events using DBSCAN on DINOv2 features'
            },
            {
                'query': 'Find similar photos',
                'params': {
                    'image_paths': ['...'],
                    'method': 'hdbscan',
                    'feature_type': 'openclip'
                },
                'description': 'Groups similar photos using HDBSCAN with semantic CLIP features'
            }
        ]
```

## Workflow Templates

### Example: Event Organization Workflow

```python
# sim_bench/agent/workflows/templates.py

from sim_bench.agent.workflows.base import Workflow, WorkflowStep, WorkflowStatus

class WorkflowTemplates:
    """Pre-defined workflow templates."""

    @staticmethod
    def organize_by_event() -> Workflow:
        """
        Organize photos by event clusters, select best from each.

        Workflow:
        1. Cluster images by visual similarity (events)
        2. For each cluster, assess quality
        3. Select top N from each cluster
        4. Generate summary report
        """
        return Workflow(
            name="organize_by_event",
            description="Organize photos into events and select best",
            steps=[
                WorkflowStep(
                    name="cluster_events",
                    tool_name="cluster_images",
                    params={
                        'method': 'dbscan',
                        'feature_type': 'dinov2',
                        'min_cluster_size': 5
                    }
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': 'clip_learned'
                    },
                    dependencies=["cluster_events"]
                ),
                WorkflowStep(
                    name="select_best",
                    tool_name="select_top_per_group",
                    params={
                        'top_n': 3
                    },
                    dependencies=["cluster_events", "assess_quality"]
                ),
                WorkflowStep(
                    name="generate_report",
                    tool_name="create_summary_report",
                    params={
                        'format': 'html'
                    },
                    dependencies=["select_best"]
                )
            ]
        )

    @staticmethod
    def find_best_portraits() -> Workflow:
        """Find and select best portrait photos."""
        return Workflow(
            name="find_best_portraits",
            description="Find portraits and select best quality",
            steps=[
                WorkflowStep(
                    name="analyze_photos",
                    tool_name="clip_tag_images",
                    params={}
                ),
                WorkflowStep(
                    name="filter_portraits",
                    tool_name="filter_by_tags",
                    params={
                        'tags': ['person', 'portrait', 'face'],
                        'min_confidence': 0.7
                    },
                    dependencies=["analyze_photos"]
                ),
                WorkflowStep(
                    name="detect_faces",
                    tool_name="detect_faces",
                    params={},
                    dependencies=["filter_portraits"]
                ),
                WorkflowStep(
                    name="assess_quality",
                    tool_name="assess_quality_batch",
                    params={
                        'method': 'nima_mobilenet'
                    },
                    dependencies=["filter_portraits"]
                ),
                WorkflowStep(
                    name="rank_portraits",
                    tool_name="rank_by_quality",
                    params={},
                    dependencies=["assess_quality", "detect_faces"]
                )
            ]
        )
```

## LLM Integration

### Prompt Templates

```python
# sim_bench/agent/llm/prompts.py

SYSTEM_PROMPT = """You are an AI photo organization assistant with access to powerful image analysis tools.

Your capabilities:
- Clustering: Group photos by visual similarity, events, or content
- Quality Assessment: Evaluate photo quality using multiple methods
- Similarity Search: Find similar images in a collection
- Photo Analysis: Tag images, detect faces, identify landmarks
- Organization: Arrange photos into hierarchical structures

Available tools:
{tool_schemas}

When a user asks to organize photos, you should:
1. Understand their intent
2. Plan a workflow using available tools
3. Execute the workflow step-by-step
4. Present results clearly
5. Offer refinements based on feedback

Always respond in structured format with:
- Understanding: Rephrase user request
- Plan: List of workflow steps
- Execution: Tool calls with parameters
- Results: Summary of findings
- Next Steps: Suggested actions
"""

PLANNING_PROMPT = """Given the user request and available tools, create a workflow plan.

User Request: {user_query}

Available Tools:
{tool_list}

Context from previous steps:
{context}

Create a workflow plan as JSON:
{{
    "workflow_name": "descriptive_name",
    "description": "what this workflow does",
    "steps": [
        {{
            "name": "step_name",
            "tool": "tool_name",
            "params": {{}},
            "reason": "why this step is needed",
            "dependencies": []
        }}
    ],
    "expected_outcome": "what user will get"
}}
"""

EXECUTION_PROMPT = """Execute the next workflow step and summarize results.

Current Step: {step_name}
Tool: {tool_name}
Parameters: {params}

Previous Results:
{previous_results}

Execute the tool and provide:
1. What was done
2. Key findings
3. Data for next steps
"""

REFINEMENT_PROMPT = """The user provided feedback on the results.

Original Request: {original_query}
Workflow Executed: {workflow_name}
Results: {results_summary}

User Feedback: {user_feedback}

Should we:
1. Adjust parameters and re-run?
2. Add additional steps?
3. Try a different approach?
4. Results are satisfactory?

Provide refined plan or confirmation.
"""
```

## Integration Example

```python
# examples/agent_demo.py

from sim_bench.agent.factory import create_agent
from sim_bench.config import GlobalConfig

def main():
    """Demo of AI agent orchestration."""

    # Load global config
    config = GlobalConfig()

    # Create agent (factory handles all initialization)
    agent = create_agent(
        llm_provider='openai',
        model='gpt-4-turbo-preview',
        config=config
    )

    # User query
    query = "Organize my vacation photos by event and select the 3 best from each event"

    # Agent plans and executes
    response = agent.process_query(query)

    # Agent returns:
    # - Understanding of request
    # - Planned workflow
    # - Execution results
    # - Suggestions for next steps

    print(response['message'])
    print(f"\nFound {response['data']['num_events']} events")
    print(f"Selected {response['data']['total_selected']} best photos")

    # Interactive refinement
    feedback = "The beach photos should be one event, not split"
    refined_response = agent.refine(feedback)
```

## Next Steps

1. Implement core agent module
2. Create tool wrappers for all capabilities
3. Build Streamlit chat interface
4. Add workflow visualization
5. Create comprehensive examples

Would you like me to start implementing these components?
