# Workflow Agent Design Document

## Overview

The Workflow Agent uses an LLM to dynamically plan and execute multi-step workflows for photo organization tasks. Unlike the Template Agent which uses pre-defined workflows, the Workflow Agent generates custom workflows based on user queries.

## Design Goals

1. **Flexibility**: Handle any user query, not just pre-defined templates
2. **Reliability**: Robust error handling and retry mechanisms
3. **Transparency**: Clear workflow visualization and step-by-step execution
4. **Efficiency**: Optimize workflow planning to minimize unnecessary steps

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
│         "Find my best vacation photos"                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Workflow Planner                            │
│  - Build prompt with available tools                    │
│  - Call LLM to generate workflow plan                   │
│  - Parse and validate workflow structure                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Workflow Builder                              │
│  - Convert LLM plan to Workflow object                  │
│  - Resolve dependencies                                 │
│  - Validate workflow DAG                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Workflow Executor                               │
│  - Execute steps in dependency order                     │
│  - Handle errors and retries                            │
│  - Store intermediate results                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Response Formatter                            │
│  - Summarize workflow execution                         │
│  - Format results for user                              │
│  - Suggest next steps                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AgentResponse                              │
└─────────────────────────────────────────────────────────┘
```

## Critical Design Improvements

### 1. Structured Workflow Plan Schema (CRITICAL)

**Problem**: LLM may generate invalid workflow plans.

**Solution**: Enforce structured schema for workflow plans.

```python
@dataclass
class WorkflowPlan:
    """Structured workflow plan from LLM."""
    name: str
    description: str
    steps: List[WorkflowStepPlan]
    estimated_duration: Optional[int] = None  # seconds
    confidence: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate plan structure."""
        errors = []
        
        if not self.name:
            errors.append("Missing workflow name")
        
        if not self.steps:
            errors.append("Workflow must have at least one step")
        
        # Validate each step
        step_names = set()
        for step in self.steps:
            if step.name in step_names:
                errors.append(f"Duplicate step name: {step.name}")
            step_names.add(step.name)
            
            # Validate dependencies exist
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Unknown dependency: {dep}")
        
        return errors

@dataclass
class WorkflowStepPlan:
    """Single step in workflow plan."""
    name: str
    tool_name: str
    params: Dict[str, Any]
    dependencies: List[str]
    reason: str  # Why this step is needed
```

### 2. Tool Result Canonicalization

**Problem**: Tool results in workflow steps can be large.

**Solution**: Store canonicalized results, reference in workflow.

```python
class WorkflowResultStorage:
    """Stores workflow step results canonically."""
    
    def store_step_result(self, step_name: str, result: Dict) -> str:
        """Store result and return reference."""
        result_id = str(uuid.uuid4())
        ref = f"workflow://{step_name}/{result_id}"
        
        # Store full result
        self.storage[ref] = {
            "step_name": step_name,
            "result": result,
            "summary": self._generate_summary(result),
            "key_data": self._extract_key_data(result)
        }
        
        return ref
```

### 3. Workflow Validation & Error Recovery

**Problem**: Invalid workflows can cause execution failures.

**Solution**: Validate and fix workflows before execution.

```python
class WorkflowValidator:
    """Validates and fixes workflow plans."""
    
    def validate_and_fix(self, plan: WorkflowPlan) -> WorkflowPlan:
        """Validate plan and attempt to fix issues."""
        errors = plan.validate()
        
        if not errors:
            return plan
        
        # Attempt to fix common issues
        fixed_plan = self._attempt_fixes(plan, errors)
        
        # Re-validate
        remaining_errors = fixed_plan.validate()
        if remaining_errors:
            raise ValueError(f"Cannot fix workflow: {remaining_errors}")
        
        return fixed_plan
    
    def _attempt_fixes(self, plan: WorkflowPlan, errors: List[str]) -> WorkflowPlan:
        """Attempt to fix workflow issues."""
        # Fix circular dependencies
        # Fix missing dependencies
        # Fix invalid tool names
        # ... fix logic
        return plan
```

## Core Components

### 1. Workflow Planner

**Purpose**: Convert user query into structured workflow plan using LLM.

**Responsibilities**:
- Build system prompt with available tools
- Format conversation context
- Call LLM with function calling or structured output
- Parse LLM response into workflow plan

**Implementation**:

```python
class WorkflowPlanner:
    """Plans workflows using LLM."""
    
    def __init__(self, llm_provider, tool_registry, memory):
        self.llm = self._create_llm_client(llm_provider)
        self.tool_registry = tool_registry
        self.memory = memory
    
    def plan_workflow(self, query: str, context: Dict) -> WorkflowPlan:
        """
        Generate workflow plan from user query.
        
        Returns:
            WorkflowPlan with steps, dependencies, and parameters
        """
        # 1. Build system prompt
        system_prompt = self._build_system_prompt(context)
        
        # 2. Get available tools
        tool_schemas = self.tool_registry.get_all_schemas()
        
        # 3. Call LLM with function calling
        llm_response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            tools=self._format_tools_for_llm(tool_schemas),
            tool_choice="auto"
        )
        
        # 4. Parse function calls into workflow steps
        workflow_steps = self._parse_function_calls(llm_response)
        
        # 5. Build workflow plan
        plan = self._build_workflow_plan(workflow_steps, context)
        
        # 6. Validate plan
        errors = plan.validate()
        if errors:
            # Attempt to fix
            plan = self.validator.validate_and_fix(plan)
        
        return plan
```

**LLM Prompt Structure**:

```
System Prompt:
You are a photo organization assistant. Your job is to plan workflows
that accomplish user goals using available tools.

Available Tools:
{tool_descriptions}

Context:
- Image paths: {image_paths}
- Previous results: {previous_results}

Workflow Planning Rules:
1. Break complex tasks into logical steps
2. Identify dependencies between steps
3. Use appropriate tools for each step
4. Minimize unnecessary steps
5. Consider parallel execution where possible

Output Format:
For each step, specify:
- Step name
- Tool to use
- Parameters
- Dependencies on previous steps
```

### 2. Workflow Builder

**Purpose**: Convert LLM plan into executable Workflow object.

**Responsibilities**:
- Validate workflow structure
- Resolve step dependencies
- Check for circular dependencies
- Optimize step ordering

**Implementation**:

```python
class WorkflowBuilder:
    """Builds Workflow objects from LLM plans."""
    
    def build_workflow(self, plan: WorkflowPlan, context: Dict) -> Workflow:
        """
        Convert workflow plan to executable Workflow.
        
        Args:
            plan: LLM-generated workflow plan
            context: User context (image_paths, etc.)
        
        Returns:
            Validated Workflow object
        """
        steps = []
        
        for step_plan in plan.steps:
            # Create WorkflowStep
            step = WorkflowStep(
                name=step_plan.name,
                tool_name=step_plan.tool_name,
                params=self._resolve_params(step_plan.params, context),
                dependencies=step_plan.dependencies
            )
            steps.append(step)
        
        # Validate workflow
        workflow = Workflow(
            name=plan.name,
            description=plan.description,
            steps=steps
        )
        
        # Check for errors
        errors = workflow.validate_dependencies()
        if errors:
            raise ValueError(f"Invalid workflow: {errors}")
        
        return workflow
```

### 3. LLM Integration Layer

**Purpose**: Abstract LLM provider differences.

**Supported Providers**:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google Gemini

**Implementation**:

```python
class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def chat(self, messages: List[Dict], tools: List[Dict] = None) -> Dict:
        """Send chat request with optional function calling."""
        pass
    
    @abstractmethod
    def format_tools(self, tool_schemas: List[Dict]) -> List[Dict]:
        """Format tool schemas for provider-specific format."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation."""
    
    def chat(self, messages, tools=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        return self._parse_response(response)
    
    def format_tools(self, tool_schemas):
        # Convert to OpenAI function calling format
        return [self._to_openai_format(schema) for schema in tool_schemas]

class GeminiProvider(LLMProvider):
    """Google Gemini implementation."""
    
    def chat(self, messages, tools=None):
        # Gemini uses different format
        response = self.client.generate_content(
            contents=messages,
            tools=tools
        )
        return self._parse_gemini_response(response)
```

## Workflow Planning Process

### Step 1: Query Analysis

```python
def analyze_query(self, query: str) -> QueryAnalysis:
    """
    Analyze user query to understand intent.
    
    Returns:
        QueryAnalysis with:
        - Intent (organize, find, analyze, etc.)
        - Entities (photos, people, events, etc.)
        - Constraints (quality, count, etc.)
    """
    # Use LLM for intent extraction
    analysis_prompt = f"""
    Analyze this query: "{query}"
    
    Extract:
    1. Primary intent
    2. Entities mentioned
    3. Constraints or requirements
    4. Expected output
    """
    
    response = self.llm.chat([{"role": "user", "content": analysis_prompt}])
    return self._parse_analysis(response)
```

### Step 2: Tool Selection

```python
def select_tools(self, analysis: QueryAnalysis) -> List[str]:
    """
    Select relevant tools based on query analysis.
    
    Returns:
        List of tool names to include in workflow planning
    """
    relevant_tools = []
    
    # Match intent to tool categories
    if analysis.intent == "organize" and "event" in analysis.entities:
        relevant_tools.extend(["cluster_images", "assess_quality_batch"])
    
    if "person" in analysis.entities:
        relevant_tools.extend(["detect_faces", "group_by_person"])
    
    # Always include quality tools if "best" mentioned
    if "best" in analysis.constraints:
        relevant_tools.append("assess_quality_batch")
    
    return relevant_tools
```

### Step 3: Workflow Generation

```python
def generate_workflow(self, query: str, selected_tools: List[str]) -> WorkflowPlan:
    """
    Generate workflow plan using LLM.
    """
    prompt = self._build_planning_prompt(query, selected_tools)
    
    # Use function calling to get structured output
    response = self.llm.chat(
        messages=[{"role": "user", "content": prompt}],
        tools=self._get_tool_schemas(selected_tools),
        tool_choice="required"  # Force tool selection
    )
    
    # Parse function calls as workflow steps
    steps = []
    for call in response.function_calls:
        step = WorkflowStepPlan(
            name=call.name,
            tool_name=call.name,
            params=call.arguments,
            dependencies=self._infer_dependencies(call, previous_steps)
        )
        steps.append(step)
    
    return WorkflowPlan(name="generated_workflow", steps=steps)
```

## Error Handling

### Error Types

1. **LLM API Errors**
   - Rate limiting
   - Timeout
   - Invalid response format
   - Token limit exceeded

2. **Workflow Planning Errors**
   - Invalid workflow structure
   - Circular dependencies
   - Missing required tools
   - Invalid parameters

3. **Execution Errors**
   - Tool execution failures
   - Dependency resolution failures
   - Resource exhaustion

### Error Recovery Strategies

```python
class WorkflowAgentErrorHandler:
    """Handles errors during workflow planning and execution."""
    
    def handle_llm_error(self, error: Exception) -> AgentResponse:
        """Handle LLM API errors."""
        if isinstance(error, RateLimitError):
            return self._retry_with_backoff()
        elif isinstance(error, TimeoutError):
            return self._fallback_to_template()
        else:
            return self._generic_error_response(error)
    
    def handle_planning_error(self, error: Exception) -> AgentResponse:
        """Handle workflow planning errors."""
        if isinstance(error, InvalidWorkflowError):
            # Try to fix workflow
            fixed_workflow = self._attempt_workflow_fix(error.workflow)
            if fixed_workflow:
                return self._execute_workflow(fixed_workflow)
            else:
                # Fallback to template matching
                return self._fallback_to_template_agent()
    
    def handle_execution_error(self, error: Exception, step: WorkflowStep) -> AgentResponse:
        """Handle workflow execution errors."""
        if step.can_retry():
            return self._retry_step(step)
        else:
            # Skip step or use alternative
            return self._handle_step_failure(step, error)
```

## State Management

### Workflow State

```python
@dataclass
class WorkflowState:
    """Tracks workflow execution state."""
    workflow: Workflow
    current_step: Optional[str]
    completed_steps: Set[str]
    failed_steps: Set[str]
    step_results: Dict[str, Any]
    execution_start: datetime
    execution_time: Optional[float]
    
    def is_complete(self) -> bool:
        return len(self.completed_steps) == len(self.workflow.steps)
    
    def get_progress(self) -> float:
        return len(self.completed_steps) / len(self.workflow.steps)
```

### Memory Integration

```python
def execute_workflow(self, workflow: Workflow) -> Dict:
    """Execute workflow with state tracking."""
    state = WorkflowState(workflow=workflow)
    
    # Store in memory
    self.memory.add_workflow_state(workflow.name, state)
    
    try:
        results = self.executor.execute_workflow(workflow)
        
        # Update state
        state.execution_time = time.time() - state.execution_start
        state.completed_steps = set(workflow.steps)
        
        # Save to memory
        self.memory.update_workflow_state(workflow.name, state)
        
        return results
    except Exception as e:
        # Save failed state
        state.failed_steps.add(state.current_step)
        self.memory.update_workflow_state(workflow.name, state)
        raise
```

## API Interface

### Public Methods

```python
class WorkflowAgent(Agent):
    """Workflow agent implementation."""
    
    def process_query(self, query: str, context: Dict = None) -> AgentResponse:
        """
        Process query by planning and executing workflow.
        
        Args:
            query: User's natural language query
            context: Optional context with image_paths, etc.
        
        Returns:
            AgentResponse with workflow results
        """
        # 1. Analyze query
        analysis = self.planner.analyze_query(query)
        
        # 2. Select tools
        tools = self.planner.select_tools(analysis)
        
        # 3. Generate workflow
        plan = self.planner.generate_workflow(query, tools)
        
        # 4. Build workflow
        workflow = self.builder.build_workflow(plan, context)
        
        # 5. Execute workflow
        results = self.executor.execute_workflow(workflow)
        
        # 6. Format response
        return self._format_response(workflow, results)
    
    def refine(self, feedback: str) -> AgentResponse:
        """
        Refine previous workflow based on feedback.
        
        Args:
            feedback: User's feedback on previous results
        
        Returns:
            AgentResponse with refined workflow results
        """
        # Get previous workflow
        previous_workflow = self.memory.get_last_workflow()
        
        # Generate refinement plan
        refinement_plan = self.planner.refine_workflow(
            previous_workflow,
            feedback
        )
        
        # Execute refined workflow
        return self.process_query(refinement_plan.query)
```

## Example Workflows

### Example 1: Simple Query

```
User: "Find my best photos"

Workflow Generated:
1. assess_quality_batch(image_paths, method='clip_learned')
2. rank_images(quality_scores)
3. select_top_images(ranked_images, top_n=10)

Execution:
✓ Step 1: Assessed quality of 500 photos
✓ Step 2: Ranked photos by quality
✓ Step 3: Selected top 10 photos

Result: "Found your 10 best photos based on quality assessment"
```

### Example 2: Complex Query

```
User: "Organize my vacation photos by location and select the 3 best from each"

Workflow Generated:
1. clip_tag_images(image_paths)  # Tag all photos
2. detect_landmarks(image_paths)  # Detect landmarks
3. group_by_location(landmarks)  # Group by location
4. assess_quality_batch(image_paths)  # Assess quality (parallel)
5. select_best_from_group(groups, quality_scores, top_n=3)  # Select best

Execution:
✓ Step 1: Tagged 200 photos
✓ Step 2: Detected landmarks in 150 photos
✓ Step 3: Grouped into 8 locations
✓ Step 4: Assessed quality of all photos
✓ Step 5: Selected 3 best from each location (24 total)

Result: "Organized 200 vacation photos into 8 locations, selected 24 best photos"
```

## Testing Strategy

### Unit Tests

```python
def test_workflow_planner():
    """Test workflow planning logic."""
    planner = WorkflowPlanner(llm_provider='mock', ...)
    
    plan = planner.plan_workflow("Find best photos", context)
    
    assert len(plan.steps) > 0
    assert all(step.tool_name for step in plan.steps)

def test_workflow_builder():
    """Test workflow building."""
    builder = WorkflowBuilder()
    
    plan = WorkflowPlan(steps=[...])
    workflow = builder.build_workflow(plan, context)
    
    assert workflow.validate_dependencies() == []
    assert len(workflow.steps) == len(plan.steps)
```

### Integration Tests

```python
def test_end_to_end_workflow():
    """Test complete workflow execution."""
    agent = create_agent(agent_type='workflow', llm_provider='mock')
    
    response = agent.process_query(
        "Find best photos",
        context={'image_paths': ['photo1.jpg', 'photo2.jpg']}
    )
    
    assert response.success
    assert 'results' in response.data
    assert len(response.data['results']) > 0
```

### Mock LLM Tests

```python
class MockLLMProvider:
    """Mock LLM for testing."""
    
    def chat(self, messages, tools=None):
        # Return predictable responses
        return {
            'function_calls': [
                {
                    'name': 'assess_quality_batch',
                    'arguments': {'image_paths': ['photo1.jpg']}
                }
            ]
        }
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: Cache workflow plans for similar queries
2. **Parallel Execution**: Execute independent steps in parallel
3. **Lazy Tool Loading**: Only load tools when needed
4. **Token Management**: Optimize prompts to reduce token usage

### Metrics to Track

- Workflow planning time
- LLM API latency
- Workflow execution time
- Success rate
- Error rate by type

## Security Considerations

1. **Input Validation**: Validate all user inputs and LLM responses
2. **Tool Parameter Validation**: Strict validation against tool schemas
3. **Rate Limiting**: Implement rate limiting for LLM calls
4. **Error Message Sanitization**: Don't expose internal errors to users

## Future Enhancements

1. **Workflow Templates**: Learn from successful workflows
2. **Multi-Agent Collaboration**: Use multiple agents for complex tasks
3. **Workflow Optimization**: Learn optimal workflows over time
4. **User Preferences**: Learn and adapt to user preferences

## Implementation Checklist

### Core Components
- [ ] LLM provider abstraction layer
- [ ] Workflow planner implementation
- [ ] Workflow builder implementation
- [ ] Error handling and recovery
- [ ] State management
- [ ] Response formatting

### Critical Improvements (MUST HAVE)
- [ ] **Structured workflow plan schema** (prevent invalid plans)
- [ ] **Tool result canonicalization** (prevent token explosion)
- [ ] **Workflow validation & fixing** (handle LLM mistakes)
- [ ] **Iterative planning** (allow LLM to refine plan)
- [ ] **Plan caching** (cache similar plans)

### Enhancements
- [ ] LLM router (model selection)
- [ ] Plan optimization
- [ ] Parallel step execution
- [ ] Workflow templates from successful plans

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Validation tests
- [ ] Error recovery tests
- [ ] Documentation
- [ ] Performance optimization


