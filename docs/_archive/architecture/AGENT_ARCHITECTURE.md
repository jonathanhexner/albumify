# Agent Architecture Explanation

## What is Template Agent?

**Template Agent** is a simple keyword-matching system that:

1. **Takes your query** (e.g., "Organize my photos by event")
2. **Matches keywords** to pre-defined workflows
3. **Executes the workflow** using tools (cluster_images, assess_quality, etc.)
4. **Returns results**

**No AI/LLM involved** - it's just `if "event" in query: use organize_by_event workflow`

## The Three Agent Types

### Template Agent (Available Now)
- **How it works**: Keyword matching → Pre-defined workflow → Execute tools
- **Pros**: Fast, no API key needed, predictable
- **Cons**: Limited to pre-defined workflows, not flexible
- **Example**: "organize by event" → matches to `organize_by_event` workflow → runs clustering + quality tools

### Workflow Agent (Not Implemented)
- **How it should work**: LLM plans custom workflow → Execute tools
- **Pros**: Flexible, handles complex queries
- **Cons**: Requires API key, slower, more expensive
- **Example**: "Find sunset photos from my vacation" → LLM creates workflow: tag images → filter sunset → filter dates → rank

### Conversational Agent (Not Implemented)
- **How it should work**: Full conversation with LLM, back-and-forth refinement
- **Pros**: Most flexible, natural language
- **Cons**: Most expensive, requires API key
- **Example**: Multi-turn conversation to refine results

## Current Workflow Error

**Error**: `Workflow blocked. Failed steps: ['cluster_by_event']`

**Why**: The `cluster_images` tool likely failed. Possible reasons:
1. Not enough images
2. Missing dependencies (DINOv2 model not downloaded)
3. Device/memory issues
4. Tool implementation bug

**How to debug**: Check the tool execution logs, check if DINOv2 is available.

## Architecture Flow

```
User Query
    ↓
TemplateAgent.process_query()
    ↓
Keyword matching → Select workflow template
    ↓
SimpleWorkflowExecutor.execute()
    ↓
For each step:
    - Get tool from registry
    - Call tool.execute() with params
    - Pass output to next step
    ↓
Return AgentResponse (success/failure + results)
```

## The Real Issue

The Template Agent is just a **proof of concept**. It's not production-ready:
- Limited error handling
- Tools may not be fully implemented
- No graceful degradation
- No user guidance on what works/doesn't work

**What you need**: Either fully implement the tools and workflows, OR build the WorkflowAgent with LLM planning.
