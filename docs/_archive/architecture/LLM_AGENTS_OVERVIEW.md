# LLM-Based Agents Overview

## Introduction

This document provides an overview of the two LLM-based agent types: **Workflow Agent** and **Conversational Agent**. Both use Large Language Models to provide intelligent photo organization capabilities, but with different interaction patterns and use cases.

## Quick Comparison

| Aspect | Workflow Agent | Conversational Agent |
|--------|---------------|---------------------|
| **Interaction** | Single query → Complete workflow | Multi-turn conversation |
| **Planning** | Plans entire workflow upfront | Plans reactively during conversation |
| **Tool Usage** | All tools in planned sequence | Tools called as needed |
| **Complexity** | Medium (~500 lines) | High (~800 lines) |
| **Use Case** | Complex, well-defined tasks | Exploration, Q&A, guidance |
| **Example** | "Organize photos by event" | "What can you do?" → "Show me faces" → "Which are best?" |

## Architecture Comparison

### Workflow Agent Flow

```
User Query
    ↓
Workflow Planner (LLM)
    ↓
Workflow Builder
    ↓
Workflow Executor
    ↓
Results
```

### Conversational Agent Flow

```
User Message
    ↓
Context Builder
    ↓
LLM (with tool calling)
    ↓
Tool Executor (if needed)
    ↓
Response Formatter
    ↓
User Response
    ↑
    └─── (Loop back for next turn)
```

## When to Use Which

### Use Workflow Agent When:

- ✅ You have a clear, complex task
- ✅ You want everything done automatically
- ✅ You need structured, reproducible workflows
- ✅ You want to minimize API calls
- ✅ Task requires multiple coordinated steps

**Example Queries:**
- "Organize my vacation photos by event and select the 3 best from each"
- "Find all photos with faces, assess quality, and group by person"
- "Cluster similar photos and select the best representative from each cluster"

### Use Conversational Agent When:

- ✅ You want to explore capabilities
- ✅ You need guidance on what's possible
- ✅ You want to refine results through conversation
- ✅ You prefer interactive, chat-like interaction
- ✅ Task requirements may change during interaction

**Example Queries:**
- "What can you do with my photos?"
- "Show me photos with faces" → "Which ones are the best quality?"
- "I have vacation photos, what should I do?"

## Shared Components

Both agents share these components:

1. **LLM Provider Abstraction**
   - Supports OpenAI, Anthropic, Gemini
   - Unified interface for all providers

2. **Tool Registry**
   - Same tools available to both agents
   - Consistent tool schemas

3. **Memory System**
   - Conversation history
   - Tool results caching
   - Context management

4. **Error Handling**
   - Common error types
   - Recovery strategies
   - User-friendly error messages

## Implementation Status

### Workflow Agent
- ✅ Architecture designed
- ✅ Gemini provider partially implemented
- ⏳ OpenAI/Anthropic providers pending
- ⏳ Workflow planner implementation pending
- ⏳ Workflow builder implementation pending

### Conversational Agent
- ✅ Architecture designed
- ⏳ All components pending implementation
- ⏳ Conversation manager pending
- ⏳ Reactive tool caller pending

## Design Documents

- **[Workflow Agent Design](./WORKFLOW_AGENT_DESIGN.md)** - Complete design for workflow-based agent
- **[Conversational Agent Design](./CONVERSATIONAL_AGENT_DESIGN.md)** - Complete design for conversational agent

## Next Steps

1. **Review Design Documents**
   - Review architecture decisions
   - Validate approach
   - Suggest improvements

2. **Implementation Priority**
   - Start with Workflow Agent (simpler)
   - Implement Gemini provider first
   - Add OpenAI/Anthropic later
   - Then implement Conversational Agent

3. **Testing Strategy**
   - Unit tests for each component
   - Integration tests for end-to-end flows
   - Mock LLM for deterministic testing

## Questions for Review

1. **Architecture**
   - Is the separation between planner, builder, and executor clear?
   - Should we share more code between agents?

2. **Error Handling**
   - Are error recovery strategies sufficient?
   - Should we add more fallback mechanisms?

3. **Performance**
   - Are optimization strategies appropriate?
   - Should we add caching layers?

4. **User Experience**
   - Are response formats user-friendly?
   - Should we add more visualizations?

5. **Security**
   - Are input validation strategies sufficient?
   - Should we add more sanitization?





