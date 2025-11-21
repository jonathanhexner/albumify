# Conversational Agent Design Document

## Overview

The Conversational Agent provides an interactive, chat-based interface for photo organization. Unlike the Workflow Agent which plans complete workflows upfront, the Conversational Agent engages in multi-turn conversations, calling tools reactively based on the conversation flow.

## Design Goals

1. **Natural Interaction**: Feel like chatting with a helpful assistant
2. **Contextual Awareness**: Remember and reference previous conversation
3. **Reactive Tool Calling**: Call tools when needed during conversation
4. **Error Recovery**: Gracefully handle errors and guide user
5. **Exploratory**: Help users discover capabilities through conversation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Conversation Loop                          │
│                                                         │
│  User Input → Context Builder → LLM → Tool Call?       │
│       ↑                                    ↓            │
│       └──────── Response Formatter ←──────┘            │
└─────────────────────────────────────────────────────────┘
```

### Detailed Flow

```
Turn 1:
User: "What can you do?"
  ↓
Context Builder: [Empty context]
  ↓
LLM: "I can organize photos, find best ones, detect faces..."
  ↓
Response: "I can help you organize photos, find your best photos..."

Turn 2:
User: "Show me photos with faces"
  ↓
Context Builder: [Previous turn + current query]
  ↓
LLM: [Decides to call detect_faces tool]
  ↓
Tool Executor: [Executes detect_faces]
  ↓
LLM: [Formats tool results into response]
  ↓
Response: "Found 45 photos with faces. Here are some examples..."

Turn 3:
User: "Which ones are the best quality?"
  ↓
Context Builder: [All previous turns + tool results]
  ↓
LLM: [Understands "ones" = photos with faces, calls assess_quality]
  ↓
Tool Executor: [Executes assess_quality on face photos]
  ↓
LLM: [Formats results]
  ↓
Response: "Here are the 10 best quality photos with faces..."
```

## Critical Design Improvements

Based on production agent patterns, the following improvements are essential for reliability:

### 1. Tool Result Canonicalization (CRITICAL)

**Problem**: Raw tool results can be large, unstructured, and cause token explosion.

**Solution**: Always normalize tool outputs with summaries and references.

```python
class ToolResultCanonicalizer:
    """Canonicalizes tool results to prevent context bloat."""
    
    def canonicalize(self, tool_name: str, result: Dict) -> Dict:
        """
        Convert raw tool result to canonical format.
        
        Returns:
            {
                "summary": "Short human-readable summary",
                "data_ref": "memory://tool_results/{tool_name}/{id}",
                "key_metrics": {...},  # Only essential numbers
                "success": bool
            }
        """
        # Generate unique reference ID
        result_id = str(uuid.uuid4())
        data_ref = f"memory://tool_results/{tool_name}/{result_id}"
        
        # Store full result in memory (not in LLM context)
        self.memory.store_tool_result(data_ref, result)
        
        # Extract key metrics only
        key_metrics = self._extract_key_metrics(result)
        
        # Generate concise summary
        summary = self._generate_summary(tool_name, result, key_metrics)
        
        return {
            "summary": summary,
            "data_ref": data_ref,
            "key_metrics": key_metrics,
            "success": result.get("success", False)
        }
    
    def _extract_key_metrics(self, result: Dict) -> Dict:
        """Extract only essential numeric metrics."""
        data = result.get("data", {})
        metrics = {}
        
        # Only include small, essential numbers
        for key in ["num_clusters", "num_images", "top_n", "num_faces"]:
            if key in data:
                metrics[key] = data[key]
        
        return metrics
    
    def _generate_summary(self, tool_name: str, result: Dict, metrics: Dict) -> str:
        """Generate concise summary (max 50 words)."""
        message = result.get("message", "")
        
        # Use LLM to summarize if message is too long
        if len(message.split()) > 50:
            summary = self._llm_summarize(message)
        else:
            summary = message
        
        # Add key metrics
        if metrics:
            metric_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            summary = f"{summary} ({metric_str})"
        
        return summary[:200]  # Hard limit
```

### 2. Context Compression with Summarization (CRITICAL)

**Problem**: Simple token trimming can cut off important information mid-sentence.

**Solution**: Semantically summarize older turns while keeping recent ones full.

```python
class ConversationCompressor:
    """Compresses conversation history intelligently."""
    
    def compress_history(self, turns: List[ConversationTurn], 
                        max_tokens: int) -> List[Dict]:
        """
        Compress conversation history with summarization.
        
        Strategy:
        - Keep last N turns full (default: 3)
        - Summarize older turns into compact notes
        - Never cut mid-sentence or mid-JSON
        """
        if len(turns) <= 3:
            return self._format_turns(turns)
        
        # Keep recent turns full
        recent_turns = turns[-3:]
        older_turns = turns[:-3]
        
        # Summarize older turns
        summary = self._summarize_turns(older_turns)
        
        # Format: summary + recent turns
        messages = [
            {
                "role": "system",
                "content": f"Earlier conversation summary: {summary}"
            }
        ]
        messages.extend(self._format_turns(recent_turns))
        
        return messages
    
    def _summarize_turns(self, turns: List[ConversationTurn]) -> str:
        """Summarize multiple turns into compact notes."""
        # Use LLM to summarize (or rule-based for simple cases)
        summary_parts = []
        
        for turn in turns:
            # Extract key information
            key_info = f"User: {turn.user_message[:50]}... "
            
            if turn.tool_calls:
                tools = [tc["name"] for tc in turn.tool_calls]
                key_info += f"Tools called: {', '.join(tools)}. "
            
            key_info += f"Result: {turn.agent_response[:50]}..."
            summary_parts.append(key_info)
        
        return " | ".join(summary_parts)
```

### 3. Structured Response Schema (CRITICAL)

**Problem**: LLM may hallucinate tool calls or produce invalid formats.

**Solution**: Force structured response format with validation.

```python
@dataclass
class StructuredAgentResponse:
    """Structured response schema for agent."""
    thought: str  # Agent's reasoning
    tool_call: Optional[Dict] = None  # Tool call if needed
    assistant_message: Optional[str] = None  # Message if no tool
    confidence: float = 1.0  # Confidence in response
    
    def validate(self) -> List[str]:
        """Validate response structure."""
        errors = []
        
        if not self.thought:
            errors.append("Missing thought")
        
        if self.tool_call and self.assistant_message:
            errors.append("Cannot have both tool_call and assistant_message")
        
        if not self.tool_call and not self.assistant_message:
            errors.append("Must have either tool_call or assistant_message")
        
        return errors

class StructuredLLMInterface:
    """Interface that enforces structured responses."""
    
    def chat_with_structure(self, messages: List[Dict]) -> StructuredAgentResponse:
        """
        Call LLM with enforced structured output.
        
        Uses JSON mode or function calling to ensure structure.
        """
        # Add structure requirement to system prompt
        structured_prompt = """
        You must respond in this exact JSON format:
        {
            "thought": "Your reasoning about what to do",
            "tool_call": {
                "name": "tool_name",
                "arguments": {...}
            } OR null,
            "assistant_message": "Message to user" OR null,
            "confidence": 0.0-1.0
        }
        
        Rules:
        - If you need to call a tool, set tool_call and assistant_message=null
        - If responding directly, set assistant_message and tool_call=null
        - Always include thought explaining your reasoning
        """
        
        # Call LLM with JSON mode
        response = self.llm.chat(
            messages + [{"role": "system", "content": structured_prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse and validate
        parsed = json.loads(response["content"])
        structured_response = StructuredAgentResponse(**parsed)
        
        # Validate
        errors = structured_response.validate()
        if errors:
            raise ValueError(f"Invalid structured response: {errors}")
        
        return structured_response
```

### 4. Iterative Tool Call Loop (CRITICAL)

**Problem**: LLM may request multiple tool calls, and tools may need to be called iteratively.

**Solution**: Implement proper iterative loop with guards.

```python
class IterativeToolCallLoop:
    """Handles iterative tool calling with safety guards."""
    
    def __init__(self, max_iterations: int = 10, max_depth: int = 5):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.call_history: List[str] = []  # Track tool calls to detect loops
    
    def execute_with_tools(self, initial_messages: List[Dict]) -> AgentResponse:
        """
        Execute conversation with iterative tool calling.
        
        Loop:
        1. Call LLM
        2. If tool calls requested → execute tools
        3. Add tool results to messages
        4. Call LLM again
        5. Repeat until no more tool calls or max iterations
        """
        messages = initial_messages
        iteration = 0
        total_tool_calls = 0
        
        while iteration < self.max_iterations:
            # Call LLM
            llm_response = self.llm.chat(messages, tools=self._get_tool_schemas())
            
            # Check for tool calls
            tool_calls = llm_response.get("tool_calls", [])
            
            if not tool_calls:
                # No more tools, return final response
                return self._format_final_response(llm_response)
            
            # Check for infinite loops
            if self._detect_loop(tool_calls):
                return self._handle_loop_detection()
            
            # Execute all tool calls
            tool_results = []
            for tool_call in tool_calls:
                # Check depth
                if total_tool_calls >= self.max_depth:
                    return self._handle_max_depth()
                
                # Execute tool
                result = self.tool_caller.handle_tool_call(tool_call, context)
                tool_results.append(result)
                
                # Track call
                call_signature = f"{tool_call['name']}({hash(str(tool_call['arguments']))})"
                self.call_history.append(call_signature)
                total_tool_calls += 1
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": self._canonicalize_result(tool_call["name"], result)
                })
            
            # Continue loop with updated messages
            iteration += 1
        
        # Max iterations reached
        return self._handle_max_iterations()
    
    def _detect_loop(self, tool_calls: List[Dict]) -> bool:
        """Detect if same tool is being called repeatedly."""
        if len(self.call_history) < 3:
            return False
        
        # Check last 3 calls
        recent = self.call_history[-3:]
        if len(set(recent)) == 1:  # All same
            return True
        
        # Check for pattern: A -> B -> A -> B
        if len(self.call_history) >= 4:
            pattern = self.call_history[-4:]
            if pattern[0] == pattern[2] and pattern[1] == pattern[3]:
                return True
        
        return False
```

### 5. Deterministic Memory References (CRITICAL)

**Problem**: Ambiguous references like `$faces` can break when context changes.

**Solution**: Use canonical memory storage with unique IDs.

```python
class CanonicalMemoryStorage:
    """Stores tool results with deterministic references."""
    
    def store_result(self, tool_name: str, result: Dict) -> str:
        """
        Store tool result and return canonical reference.
        
        Returns:
            Reference ID like "mem://detect_faces/abc123"
        """
        # Generate deterministic ID from tool + timestamp + hash
        result_id = self._generate_id(tool_name, result)
        ref = f"mem://{tool_name}/{result_id}"
        
        # Store full result
        self.storage[ref] = {
            "tool_name": tool_name,
            "result": result,
            "summary": self._generate_summary(result),
            "timestamp": datetime.now(),
            "key_data": self._extract_key_data(result)
        }
        
        return ref
    
    def resolve_reference(self, ref: str) -> Dict:
        """Resolve canonical reference to actual data."""
        if not ref.startswith("mem://"):
            raise ValueError(f"Invalid reference format: {ref}")
        
        stored = self.storage.get(ref)
        if not stored:
            raise KeyError(f"Reference not found: {ref}")
        
        return stored
    
    def list_available_references(self, tool_name: str = None) -> List[str]:
        """List available references, optionally filtered by tool."""
        refs = list(self.storage.keys())
        
        if tool_name:
            refs = [r for r in refs if f"/{tool_name}/" in r]
        
        return refs
```

## Core Components

### 1. Conversation Manager

**Purpose**: Manages multi-turn conversation state and flow.

**Responsibilities**:
- Track conversation history
- Manage context across turns
- Handle conversation state transitions
- Detect conversation end

**Implementation**:

```python
class ConversationManager:
    """Manages conversation state and flow."""
    
    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self.current_conversation_id = None
        self.turn_count = 0
    
    def start_conversation(self, initial_query: str) -> str:
        """Start new conversation."""
        self.current_conversation_id = str(uuid.uuid4())
        self.turn_count = 0
        
        self.memory.start_conversation(
            conversation_id=self.current_conversation_id,
            initial_query=initial_query
        )
        
        return self.current_conversation_id
    
    def add_turn(self, user_message: str, agent_response: str, 
                 tool_calls: List[Dict] = None):
        """Add conversation turn."""
        self.turn_count += 1
        
        turn = ConversationTurn(
            turn_number=self.turn_count,
            user_message=user_message,
            agent_response=agent_response,
            tool_calls=tool_calls or [],
            timestamp=datetime.now()
        )
        
        self.memory.add_turn(self.current_conversation_id, turn)
    
    def get_conversation_context(self, max_turns: int = 10) -> List[Dict]:
        """Get conversation context for LLM."""
        turns = self.memory.get_recent_turns(
            self.current_conversation_id,
            max_turns
        )
        
        # Format for LLM
        messages = []
        for turn in turns:
            messages.append({
                "role": "user",
                "content": turn.user_message
            })
            
            # Add tool calls if any
            if turn.tool_calls:
                for tool_call in turn.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": tool_call["result"]
                    })
            
            messages.append({
                "role": "assistant",
                "content": turn.agent_response
            })
        
        return messages
```

### 2. Reactive Tool Caller

**Purpose**: Execute tools reactively based on LLM decisions during conversation.

**Responsibilities**:
- Parse LLM tool call requests
- Validate tool calls
- Execute tools
- Format tool results for LLM
- Handle tool errors

**Implementation**:

```python
class ReactiveToolCaller:
    """Handles reactive tool calling during conversation."""
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    def handle_tool_call(self, tool_call: Dict, context: Dict) -> Dict:
        """
        Execute tool call and return formatted result.
        
        Args:
            tool_call: LLM tool call request
            context: Current conversation context
        
        Returns:
            Formatted tool result for LLM
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        # Validate tool exists
        if not self.tool_registry.has_tool(tool_name):
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": self.tool_registry.list_tools()
            }
        
        # Get tool
        tool = self.tool_registry.get_tool(tool_name)
        
        # Validate arguments
        try:
            tool.validate_params(**arguments)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}
        
        # Resolve context-dependent parameters
        resolved_args = self._resolve_parameters(arguments, context)
        
        # Execute tool
        try:
            result = tool.execute(**resolved_args)
            
            # Format result for LLM
            return self._format_tool_result(tool_name, result)
            
        except Exception as e:
            return {
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name
            }
    
    def _resolve_parameters(self, arguments: Dict, context: Dict) -> Dict:
        """Resolve parameters that reference conversation context."""
        resolved = {}
        
        for key, value in arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous result
                ref_path = value[1:]  # Remove $
                resolved[key] = self._resolve_reference(ref_path, context)
            else:
                resolved[key] = value
        
        return resolved
    
    def _format_tool_result(self, tool_name: str, result: Dict) -> str:
        """
        Format tool result for LLM consumption using canonical format.
        
        CRITICAL: Never return raw results - always canonicalize.
        """
        # Canonicalize result first
        canonical = self.canonicalizer.canonicalize(tool_name, result)
        
        if not canonical.get("success"):
            return f"Tool '{tool_name}' failed: {canonical.get('summary', 'Unknown error')}"
        
        # Return only summary and reference
        formatted = f"Tool '{tool_name}' executed successfully.\n"
        formatted += f"Summary: {canonical['summary']}\n"
        formatted += f"Reference: {canonical['data_ref']}\n"
        
        # Add key metrics only
        if canonical.get("key_metrics"):
            metrics = canonical["key_metrics"]
            formatted += f"Key metrics: {metrics}\n"
        
        return formatted
```

### 3. Context Builder

**Purpose**: Build comprehensive context for LLM from conversation history and available data.

**Responsibilities**:
- Aggregate conversation history
- Include relevant tool results
- Manage token limits
- Format context for LLM

**Implementation**:

```python
class ConversationContextBuilder:
    """Builds context for conversational LLM."""
    
    def __init__(self, memory: AgentMemory, tool_registry: ToolRegistry):
        self.memory = memory
        self.tool_registry = tool_registry
        self.max_context_tokens = 8000  # Leave room for response
    
    def build_context(self, conversation_id: str, current_query: str) -> List[Dict]:
        """
        Build conversation context for LLM.
        
        Returns:
            List of messages formatted for LLM
        """
        # Get conversation history
        history = self.memory.get_conversation_history(conversation_id)
        
        # Get available tools
        tool_schemas = self.tool_registry.get_all_schemas()
        
        # Get previous tool results
        previous_results = self.memory.get_recent_tool_results(conversation_id)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(tool_schemas, previous_results)
        
        # Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (with token management)
        history_messages = self._format_history(history)
        messages.extend(history_messages)
        
        # Add current query
        messages.append({"role": "user", "content": current_query})
        
        # Check token limit and trim if needed
        total_tokens = self._estimate_tokens(messages)
        if total_tokens > self.max_context_tokens:
            messages = self._trim_to_fit(messages, self.max_context_tokens)
        
        return messages
    
    def _build_system_prompt(self, tool_schemas: List[Dict], 
                            previous_results: Dict) -> str:
        """
        Build system prompt with persona, constraints, and tools.
        
        CRITICAL: Include strict constraints to prevent hallucinations.
        """
        prompt = """You are a helpful photo organization assistant. 
You can help users organize, analyze, and manage their photos through conversation.

CRITICAL CONSTRAINTS:
- Never ask user factual questions you can answer from tools
- Never invent filenames or paths - always use actual data
- Never assume file paths exist - always validate first
- Always validate tool parameters before calling tools
- Reference previous results using their memory references (mem://...)
- If unsure, use tools to get information rather than guessing

Available Tools:
"""
        # Add tool descriptions
        for schema in tool_schemas:
            prompt += f"- {schema['name']}: {schema['description']}\n"
        
        # Add previous results as references only
        if previous_results:
            prompt += "\nPrevious Results Available (use references):\n"
            for ref, summary in previous_results.items():
                prompt += f"- {ref}: {summary}\n"
        
        prompt += """
Response Format:
You must respond in this JSON structure:
{
    "thought": "Your reasoning about what to do",
    "tool_call": {"name": "...", "arguments": {...}} OR null,
    "assistant_message": "Message to user" OR null,
    "confidence": 0.0-1.0
}

Guidelines:
- Be conversational and helpful
- Call tools when needed to answer user questions
- Reference previous results using their memory references
- Ask clarifying questions only when truly needed
- Provide clear, actionable responses
- Always include your reasoning in "thought"
"""
        return prompt
    
    def _trim_to_fit(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Trim messages to fit token limit using semantic compression.
        
        CRITICAL: Never cut mid-sentence or mid-JSON. Use summarization.
        """
        # Keep system prompt
        system_msg = messages[0]
        remaining = messages[1:]
        
        # Separate into recent and old
        if len(remaining) <= 3:
            return [system_msg] + remaining
        
        recent = remaining[-3:]  # Keep last 3 full
        older = remaining[:-3]   # Summarize older
        
        # Summarize older messages semantically
        summary = self._summarize_messages(older)
        
        # Build compressed messages
        compressed = [system_msg]
        
        # Add summary of older messages
        if summary:
            compressed.append({
                "role": "system",
                "content": f"Earlier conversation (summarized): {summary}"
            })
        
        # Add recent messages
        compressed.extend(recent)
        
        # Verify token count
        total_tokens = self._estimate_tokens(compressed)
        if total_tokens > max_tokens:
            # Further compress recent messages if needed
            compressed = self._aggressive_compress(compressed, max_tokens)
        
        return compressed
    
    def _summarize_messages(self, messages: List[Dict]) -> str:
        """Semantically summarize older messages."""
        # Extract key information from each message
        summaries = []
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if role == "user":
                summaries.append(f"User: {content[:100]}...")
            elif role == "assistant":
                summaries.append(f"Assistant: {content[:100]}...")
            elif role == "tool":
                # Tool results are already canonicalized (short)
                summaries.append(f"Tool result: {content[:50]}...")
        
        return " | ".join(summaries)
```

### 4. Response Formatter

**Purpose**: Format LLM responses for user consumption.

**Responsibilities**:
- Format tool results into user-friendly messages
- Generate suggestions for next steps
- Create visualizations when appropriate
- Handle error messages gracefully

**Implementation**:

```python
class ConversationResponseFormatter:
    """Formats LLM responses for users."""
    
    def format_response(self, llm_response: Dict, tool_results: List[Dict] = None) -> AgentResponse:
        """
        Format LLM response into AgentResponse.
        
        Args:
            llm_response: Raw LLM response
            tool_results: Results from tool calls (if any)
        
        Returns:
            Formatted AgentResponse
        """
        # Extract text response
        message = llm_response.get("content", "")
        
        # Extract tool calls if any
        tool_calls = llm_response.get("tool_calls", [])
        
        # Build response data
        data = {
            "message": message,
            "tool_calls": tool_calls
        }
        
        # Add tool results if available
        if tool_results:
            data["tool_results"] = [
                self._format_tool_result(tr) for tr in tool_results
            ]
        
        # Generate suggestions
        suggestions = self._generate_suggestions(llm_response, tool_results)
        if suggestions:
            data["suggestions"] = suggestions
        
        return AgentResponse(
            success=True,
            message=message,
            data=data,
            metadata={
                "tokens_used": llm_response.get("usage", {}).get("total_tokens", 0),
                "tool_calls_count": len(tool_calls)
            },
            next_steps=suggestions
        )
    
    def _generate_suggestions(self, llm_response: Dict, 
                            tool_results: List[Dict]) -> List[str]:
        """Generate helpful suggestions for next steps."""
        suggestions = []
        
        # If tools were called, suggest follow-ups
        if tool_results:
            for result in tool_results:
                tool_name = result.get("tool_name")
                
                if tool_name == "detect_faces":
                    suggestions.append("Would you like to group these photos by person?")
                    suggestions.append("Want to see the best quality photos with faces?")
                
                elif tool_name == "cluster_images":
                    suggestions.append("Would you like to select the best photo from each cluster?")
                    suggestions.append("Want to see photos from a specific cluster?")
        
        return suggestions[:3]  # Limit to 3 suggestions
```

## Conversation Flow Patterns

### Pattern 1: Information Request

```
User: "What can you do?"
Agent: [No tool call, just responds with capabilities]
User: "Show me an example"
Agent: [Calls example tool or provides example]
```

### Pattern 2: Task Execution

```
User: "Find photos with faces"
Agent: [Calls detect_faces tool]
Agent: "Found 45 photos with faces"
User: "Which are the best quality?"
Agent: [Calls assess_quality on those 45 photos]
Agent: "Here are the 10 best quality photos with faces"
```

### Pattern 3: Exploration

```
User: "What photos do I have?"
Agent: [Calls list_images or analyzes directory]
Agent: "You have 200 photos. Would you like me to analyze them?"
User: "Yes, group them by event"
Agent: [Calls cluster_images]
Agent: "Grouped into 8 events. Want to see them?"
```

### Pattern 4: Refinement

```
User: "Find my best photos"
Agent: [Calls assess_quality, rank_images]
Agent: "Found your top 20 best photos"
User: "But only from vacation"
Agent: [Calls filter_by_tags with 'vacation', then re-ranks]
Agent: "Here are your 12 best vacation photos"
```

## Error Handling

### Error Types

1. **Ambiguous Queries**
   - User: "Do that"
   - Solution: Ask for clarification

2. **Tool Execution Failures**
   - Solution: Explain error, suggest alternatives

3. **Context Loss**
   - Solution: Re-ask or summarize context

4. **Token Limit Exceeded**
   - Solution: Summarize old conversation, continue

### Error Recovery

```python
class ConversationErrorHandler:
    """Handles errors during conversation."""
    
    def handle_ambiguous_query(self, query: str, context: Dict) -> AgentResponse:
        """Handle ambiguous user query."""
        # Ask for clarification
        clarification_prompt = f"""
        The user said: "{query}"
        
        This is ambiguous. Based on conversation history, what might they mean?
        Ask a clarifying question.
        """
        
        response = self.llm.chat([{"role": "user", "content": clarification_prompt}])
        
        return AgentResponse(
            success=False,
            message=response["content"],
            data={"needs_clarification": True}
        )
    
    def handle_tool_error(self, error: Exception, tool_name: str) -> str:
        """Format tool error for LLM."""
        error_message = f"""
        Tool '{tool_name}' failed with error: {str(error)}
        
        How should I respond to the user? Should I:
        1. Try an alternative approach?
        2. Ask the user for more information?
        3. Explain what went wrong?
        """
        
        response = self.llm.chat([{"role": "user", "content": error_message}])
        return response["content"]
```

## State Management

### Conversation State

```python
@dataclass
class ConversationState:
    """Tracks conversation state."""
    conversation_id: str
    turns: List[ConversationTurn]
    active_tool_calls: List[str]  # Tools currently executing
    available_data: Dict[str, Any]  # Results from previous tool calls
    user_intent: Optional[str]  # Inferred user intent
    conversation_phase: str  # 'exploration', 'execution', 'refinement'
    
    def get_summary(self) -> str:
        """Get conversation summary for context."""
        summary = f"Conversation has {len(self.turns)} turns.\n"
        
        if self.user_intent:
            summary += f"User intent: {self.user_intent}\n"
        
        if self.available_data:
            summary += f"Available data: {list(self.available_data.keys())}\n"
        
        return summary
```

## API Interface

### Public Methods

```python
class ConversationalAgent(Agent):
    """Conversational agent implementation."""
    
    def __init__(self, llm_provider, tool_registry, memory, config):
        super().__init__(config)
        self.llm = self._create_llm_client(llm_provider)
        self.tool_registry = tool_registry
        self.memory = memory
        self.conversation_manager = ConversationManager(memory)
        self.tool_caller = ReactiveToolCaller(tool_registry)
        self.context_builder = ConversationContextBuilder(memory, tool_registry)
        self.response_formatter = ConversationResponseFormatter()
    
    def process_query(self, query: str, context: Dict = None) -> AgentResponse:
        """
        Process query in conversational context.
        
        Args:
            query: User's message
            context: Optional context (conversation_id, image_paths, etc.)
        
        Returns:
            AgentResponse with conversational response
        """
        # Get or create conversation
        conversation_id = context.get("conversation_id") or \
                         self.conversation_manager.start_conversation(query)
        
        # Build context
        messages = self.context_builder.build_context(conversation_id, query)
        
        # Call LLM
        llm_response = self.llm.chat(messages, tools=self._get_tool_schemas())
        
        # Handle tool calls iteratively (CRITICAL: Support multiple tool calls)
        tool_results = []
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            # Check for tool calls
            tool_calls = llm_response.get("tool_calls", [])
            
            if not tool_calls:
                # No more tools needed, break loop
                break
            
            # Check for infinite loops
            if self._detect_tool_loop(tool_calls, tool_results):
                logger.warning("Detected potential infinite loop in tool calls")
                break
            
            # Execute all tool calls
            for tool_call in tool_calls:
                result = self.tool_caller.handle_tool_call(tool_call, context)
                tool_results.append(result)
                
                # Canonicalize result before adding to context
                canonical = self.canonicalizer.canonicalize(
                    tool_call["name"],
                    result
                )
                
                # Add canonicalized result to conversation
                messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": canonical["summary"] + f" (ref: {canonical['data_ref']})"
                })
            
            # Call LLM again with tool results
            llm_response = self.llm.chat(messages, tools=self._get_tool_schemas())
            iteration += 1
        
        if iteration >= max_iterations:
            logger.warning(f"Reached max iterations ({max_iterations}) in tool call loop")
        
        # Format response
        response = self.response_formatter.format_response(
            llm_response,
            tool_results
        )
        
        # Save turn
        self.conversation_manager.add_turn(
            query,
            response.message,
            llm_response.get("tool_calls", [])
        )
        
        return response
    
    def refine(self, feedback: str) -> AgentResponse:
        """
        Refine based on user feedback.
        
        Args:
            feedback: User's feedback on previous response
        
        Returns:
            AgentResponse with refined approach
        """
        # Get last conversation turn
        last_turn = self.memory.get_last_turn()
        
        # Build refinement context
        refinement_prompt = f"""
        Previous interaction:
        User: {last_turn.user_message}
        Assistant: {last_turn.agent_response}
        
        User feedback: {feedback}
        
        How should I adjust my approach?
        """
        
        # Process as new query with refinement context
        return self.process_query(feedback, {
            "refinement_context": refinement_prompt
        })
```

## Example Conversations

### Example 1: Exploration

```
User: "What can you do with my photos?"
Agent: "I can help you organize photos, find your best ones, detect faces and group by person, identify landmarks, assess quality, and more. What would you like to do?"

User: "I have vacation photos, what should I do?"
Agent: "For vacation photos, I can:
- Organize them by event or location
- Find your best photos
- Group photos with people
- Identify landmarks and places
Which would be most helpful?"

User: "Organize by event"
Agent: [Calls cluster_images tool]
Agent: "I've organized your 200 vacation photos into 8 events. Would you like to see the best photos from each event?"
```

### Example 2: Task Execution

```
User: "Find photos with faces"
Agent: [Calls detect_faces]
Agent: "Found 45 photos with faces. Here are some examples: [shows thumbnails]"

User: "Which ones are the best quality?"
Agent: [Calls assess_quality_batch on the 45 photos]
Agent: "Here are the 10 best quality photos with faces: [shows results]"

User: "Group them by person"
Agent: [Calls group_by_person on the 45 photos]
Agent: "Grouped into 5 people. Here's the breakdown: [shows groups]"
```

## Testing Strategy

### Unit Tests

```python
def test_conversation_manager():
    """Test conversation state management."""
    manager = ConversationManager(memory)
    
    conv_id = manager.start_conversation("Hello")
    assert conv_id is not None
    
    manager.add_turn("Hello", "Hi there!")
    context = manager.get_conversation_context()
    assert len(context) > 0

def test_tool_caller():
    """Test reactive tool calling."""
    caller = ReactiveToolCaller(tool_registry)
    
    tool_call = {"name": "detect_faces", "arguments": {"image_paths": ["photo.jpg"]}}
    result = caller.handle_tool_call(tool_call, {})
    
    assert "success" in result or "error" in result
```

### Integration Tests

```python
def test_conversational_flow():
    """Test complete conversational flow."""
    agent = create_agent(agent_type='conversational', llm_provider='mock')
    
    # Turn 1
    response1 = agent.process_query("What can you do?")
    assert response1.success
    
    # Turn 2 (with conversation context)
    response2 = agent.process_query(
        "Find photos with faces",
        context={"conversation_id": response1.metadata["conversation_id"]}
    )
    assert response2.success
    assert "tool_calls" in response2.data or "faces" in response2.message.lower()
```

## Additional Critical Components

### 6. LLM Router (Model Selection)

**Purpose**: Route requests to appropriate model based on task complexity.

```python
class LLMRouter:
    """Routes requests to appropriate LLM model."""
    
    def __init__(self):
        self.small_model = "gpt-3.5-turbo"  # For simple tasks
        self.main_model = "gpt-4-turbo"      # For complex reasoning
        self.cheap_model = "gemini-pro"      # For high-volume tasks
    
    def select_model(self, query: str, context: Dict) -> str:
        """
        Select appropriate model based on task complexity.
        
        Returns:
            Model identifier
        """
        # Simple extraction tasks → small model
        if self._is_simple_extraction(query):
            return self.small_model
        
        # High-volume summarization → cheap model
        if self._is_summarization(query):
            return self.cheap_model
        
        # Complex reasoning → main model
        return self.main_model
    
    def _is_simple_extraction(self, query: str) -> bool:
        """Check if query is simple extraction."""
        simple_patterns = [
            "what is", "how many", "list", "show me",
            "extract", "get", "find all"
        ]
        return any(pattern in query.lower() for pattern in simple_patterns)
    
    def _is_summarization(self, query: str) -> bool:
        """Check if query is summarization task."""
        return "summarize" in query.lower() or "summary" in query.lower()
```

### 7. Planning Mode

**Purpose**: Allow LLM to create a light plan before executing.

```python
class PlanningMode:
    """Light planning mode for complex tasks."""
    
    def create_plan(self, query: str, context: Dict) -> Dict:
        """
        Create lightweight plan before execution.
        
        Returns:
            {
                "plan": ["step1", "step2", "step3"],
                "estimated_tools": ["tool1", "tool2"],
                "confidence": 0.0-1.0
            }
        """
        planning_prompt = f"""
        User request: {query}
        
        Create a brief plan:
        1. What tools will be needed?
        2. What order should they be called?
        3. What are the dependencies?
        
        Respond in JSON:
        {{
            "plan": ["step1", "step2", ...],
            "estimated_tools": ["tool1", "tool2", ...],
            "confidence": 0.0-1.0
        }}
        """
        
        response = self.llm.chat([{"role": "user", "content": planning_prompt}])
        return json.loads(response["content"])
```

### 8. Throttling & Rate Limiting

**Purpose**: Prevent API quota exhaustion and control costs.

```python
class ThrottleManager:
    """Manages throttling for LLM and tool calls."""
    
    def __init__(self, config: Dict):
        self.max_calls_per_minute = config.get("max_calls_per_minute", 60)
        self.max_tokens_per_hour = config.get("max_tokens_per_hour", 100000)
        self.call_history: List[datetime] = []
        self.token_usage: List[int] = []
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit allows another call."""
        now = datetime.now()
        
        # Remove old entries
        self.call_history = [
            ts for ts in self.call_history
            if (now - ts).seconds < 60
        ]
        
        return len(self.call_history) < self.max_calls_per_minute
    
    def record_call(self, tokens_used: int):
        """Record API call for rate limiting."""
        self.call_history.append(datetime.now())
        self.token_usage.append(tokens_used)
        
        # Trim old token usage (keep last hour)
        hour_ago = datetime.now() - timedelta(hours=1)
        # ... trim logic
    
    def should_throttle(self) -> bool:
        """Check if throttling is needed."""
        return not self.check_rate_limit() or self._exceeds_token_limit()
```

## Performance Considerations

### Optimization Strategies

1. **Context Caching**: Cache formatted context for recent turns
2. **Lazy Tool Loading**: Only load tools when called
3. **Parallel Tool Execution**: Execute independent tools in parallel
4. **Conversation Summarization**: Summarize old turns to save tokens (CRITICAL)
5. **Model Routing**: Use cheaper models for simple tasks (cost reduction)
6. **Result Canonicalization**: Store only summaries in context (token reduction)

### Metrics to Track

- Average conversation length
- Tool calls per conversation
- Response time per turn
- Token usage per conversation
- User satisfaction
- **Tool call loop iterations** (detect inefficiencies)
- **Context compression ratio** (how much was summarized)
- **Model routing accuracy** (did we use the right model?)

## Security Considerations

1. **Input Sanitization**: Sanitize all user inputs
2. **Tool Parameter Validation**: Strict validation before execution
3. **Context Privacy**: Don't expose sensitive data in prompts
4. **Rate Limiting**: Limit conversation length and tool calls

## Future Enhancements

1. **Conversation Memory**: Long-term memory across sessions
2. **User Preferences**: Learn and remember user preferences
3. **Multi-Modal**: Support image inputs in conversation
4. **Voice Interface**: Voice input/output support

## Implementation Checklist

### Core Components
- [ ] Conversation manager
- [ ] Reactive tool caller
- [ ] Context builder with token management
- [ ] Response formatter
- [ ] Error handling
- [ ] State management
- [ ] LLM integration

### Critical Improvements (MUST HAVE)
- [ ] **Tool result canonicalization** (prevent token explosion)
- [ ] **Context compression with summarization** (semantic, not just trimming)
- [ ] **Structured response schema** (prevent hallucinations)
- [ ] **Iterative tool call loop** (handle multiple tool calls)
- [ ] **Infinite loop detection** (prevent runaway costs)
- [ ] **Deterministic memory references** (canonical storage)
- [ ] **Persona & constraints in system prompt** (reduce errors)

### Enhancements (SHOULD HAVE)
- [ ] LLM router (model selection)
- [ ] Planning mode (light planning)
- [ ] Throttling & rate limiting
- [ ] Conversation summarization
- [ ] Parallel tool execution

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Loop detection tests
- [ ] Token limit tests
- [ ] Error recovery tests
- [ ] Documentation

## Critical Implementation Notes

### 1. Always Canonicalize Tool Results

```python
# ❌ WRONG: Raw result in context
messages.append({
    "role": "tool",
    "content": str(large_result_dict)  # Could be 10,000 tokens!
})

# ✅ CORRECT: Canonicalized result
canonical = canonicalizer.canonicalize(tool_name, result)
messages.append({
    "role": "tool",
    "content": canonical["summary"] + f" (ref: {canonical['data_ref']})"
})
```

### 2. Always Use Iterative Loop for Tool Calls

```python
# ❌ WRONG: Single tool call assumption
if tool_calls:
    execute_tools()
    get_final_response()

# ✅ CORRECT: Iterative loop
while has_tool_calls and iteration < max_iterations:
    execute_tools()
    get_llm_response()  # May request more tools
    iteration += 1
```

### 3. Always Compress Context Semantically

```python
# ❌ WRONG: Simple truncation
messages = messages[-10:]  # Cuts mid-sentence!

# ✅ CORRECT: Semantic compression
recent = messages[-3:]  # Keep full
older = messages[:-3]   # Summarize
compressed = [summarize(older)] + recent
```

### 4. Always Enforce Structured Responses

```python
# ❌ WRONG: Free-form LLM response
response = llm.chat(messages)  # May hallucinate!

# ✅ CORRECT: Structured schema
response = llm.chat(messages, response_format={"type": "json_object"})
structured = StructuredAgentResponse(**json.loads(response))
structured.validate()  # Ensure valid
```


