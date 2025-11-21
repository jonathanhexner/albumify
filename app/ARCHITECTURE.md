# Photo Organization App - Production Architecture

## Overview

This is a **production-grade Streamlit application** built with clean architecture principles, proper separation of concerns, and full type safety.

## Architecture Principles

1. **Separation of Concerns**: Each layer has a single, well-defined responsibility
2. **Dependency Inversion**: Business logic is framework-agnostic
3. **Modularity**: Components are loosely coupled and highly cohesive
4. **Testability**: Core logic can be tested without UI framework
5. **Type Safety**: Full type hints throughout
6. **Clear Boundaries**: Well-defined interfaces between layers

## Directory Structure

```
app/
├── main.py                     # Entry point (minimal - just wiring)
├── __init__.py                 # Package exports
│
├── config/                     # Configuration Layer
│   ├── __init__.py            # Exports: Settings, AgentType, ImageFormat
│   ├── constants.py           # Enums, constants, example queries
│   └── settings.py            # Settings dataclasses (AppConfig, AgentConfig)
│
├── core/                       # Business Logic Layer (Framework-Agnostic)
│   ├── __init__.py            # Exports: models, services, exceptions
│   ├── models.py              # Domain models (AppState, ImageLibrary, etc.)
│   ├── services.py            # Business logic (AgentService, ImageService)
│   └── exceptions.py          # Custom exceptions (AgentError, ImageLoadError)
│
├── state/                      # State Management Layer
│   ├── __init__.py            # Exports: StateManager
│   └── manager.py             # State manager (bridges Streamlit session state)
│
└── ui/                         # Presentation Layer (Streamlit-Specific)
    ├── __init__.py            # Exports: render_app
    ├── pages.py               # Page orchestration
    ├── styles.py              # CSS styling
    └── components/            # Reusable UI components
        ├── __init__.py        # Component exports
        ├── sidebar.py         # Sidebar component
        ├── chat.py            # Chat interface component
        └── status.py          # Status display component
```

## Layer Responsibilities

### 1. Configuration Layer (`config/`)

**Purpose**: Centralized configuration and constants

**Contents**:
- `constants.py`: Enums (AgentType, ImageFormat), constants, example queries
- `settings.py`: Configuration dataclasses (AppConfig, AgentConfig, Settings)

**Characteristics**:
- Read-only configuration
- No business logic
- Can be imported by any layer

### 2. Core Layer (`core/`)

**Purpose**: Framework-agnostic business logic

**Components**:

- **`models.py`**: Domain models
  - `AppState`: Complete application state
  - `ImageLibrary`: Loaded image collection
  - `ChatMessage`: Single chat message
  - `AgentInstance`: Initialized agent wrapper
  - `AgentStatus`, `MessageRole`: Enums

- **`services.py`**: Business logic services
  - `AgentService`: Agent operations (initialize, process_query)
  - `ImageService`: Image operations (load_directory, validate)
  - `ConversationService`: Message creation and formatting

- **`exceptions.py`**: Custom exceptions
  - `AgentError`: Agent operation failures
  - `ImageLoadError`: Image loading failures
  - `ValidationError`: Input validation failures

**Characteristics**:
- No knowledge of Streamlit
- Pure Python business logic
- Fully testable without UI
- Returns domain models, not UI components

### 3. State Layer (`state/`)

**Purpose**: Bridge between Streamlit session state and domain models

**Component**:
- `StateManager`: Clean interface to Streamlit session state
  - Stores `AppState` in session
  - Provides typed getters/setters
  - Encapsulates session state details

**Characteristics**:
- Single source of truth for app state
- Type-safe interface
- Hides Streamlit session state implementation
- All state access goes through manager

### 4. UI Layer (`ui/`)

**Purpose**: Streamlit-specific presentation code

**Components**:

- **`pages.py`**: Page orchestration
  - `render_app()`: Main entry point
  - Configures Streamlit
  - Coordinates components

- **`styles.py`**: CSS styling
  - `get_custom_css()`: Returns custom CSS

- **`components/`**: Reusable UI components
  - `sidebar.py`: Complete sidebar (agent init, image loading, controls)
  - `chat.py`: Chat interface (conversation, input, examples)
  - `status.py`: Status display (agent, images)

**Characteristics**:
- Only layer that knows about Streamlit
- Pure presentation logic
- Calls services for business logic
- Uses StateManager for state

## Data Flow

### Typical User Action Flow

```
User clicks "Initialize Agent"
    ↓
sidebar.py: _handle_agent_initialization()
    ↓
AgentService.initialize_agent() → Returns AgentInstance
    ↓
StateManager.set_agent(agent_instance)
    ↓
st.rerun() → UI updates
```

### Query Processing Flow

```
User types query → chat.py: _handle_user_query()
    ↓
ConversationService.create_message() → Creates ChatMessage
    ↓
StateManager.add_message(user_message)
    ↓
StateManager.get_agent() → Get AgentInstance
StateManager.get_context() → Get context dict
    ↓
AgentService.process_query() → Returns AgentResponse
    ↓
ConversationService.format_*_message() → Format response
    ↓
StateManager.add_message(assistant_message)
    ↓
st.rerun() → UI shows new messages
```

## Key Design Patterns

### 1. Service Layer Pattern
- Business logic in services (`AgentService`, `ImageService`)
- Services are stateless
- Clear input/output contracts

### 2. Repository Pattern
- `StateManager` abstracts state storage
- Domain models for data
- UI doesn't touch session state directly

### 3. Dependency Inversion
- Core layer defines interfaces
- UI layer implements/uses them
- Core has no dependency on UI

### 4. Single Responsibility
- Each module has one clear purpose
- Each class has one reason to change

### 5. Type Safety
- Dataclasses for all models
- Type hints everywhere
- Enums for fixed sets

## Error Handling

### Exception Hierarchy

```
AppError (base)
├── AgentError (agent operations)
├── ImageLoadError (image loading)
└── ValidationError (input validation)
```

### Error Handling Strategy

1. **Services raise custom exceptions**
   - `AgentService` raises `AgentError`
   - `ImageService` raises `ImageLoadError`
   - Both validate inputs and raise `ValidationError`

2. **UI catches and displays**
   - Try/except in UI handlers
   - Format user-friendly messages
   - Log technical details

3. **No silent failures**
   - All errors are logged
   - All errors shown to user
   - Stack traces in logs

## Testing Strategy

### Unit Testing Core Layer

```python
# core/ is framework-agnostic - easy to test

def test_agent_service_initialization():
    agent = AgentService.initialize_agent(AgentType.TEMPLATE)
    assert agent.status == AgentStatus.READY

def test_image_service_validation():
    is_valid, msg = ImageService.validate_directory("/invalid")
    assert not is_valid
    assert "not found" in msg.lower()
```

### Integration Testing UI Layer

```python
# UI layer tests require Streamlit test framework
from streamlit.testing.v1 import AppTest

def test_full_app_flow():
    app = AppTest.from_file("app/main.py")
    app.run()
    # Test interactions...
```

## Extension Points

### Adding a New Agent Type

1. Add to `config/constants.py`: `AgentType.NEW`
2. Add to `config/settings.py`: `Settings.AGENTS`
3. Implement in `sim_bench.agent.factory`
4. No UI changes needed!

### Adding a New Feature

1. **Add domain model** in `core/models.py` (if needed)
2. **Add service method** in `core/services.py`
3. **Add state methods** in `state/manager.py` (if needed)
4. **Add UI component** in `ui/components/`
5. **Wire in** `ui/pages.py`

### Adding Configuration

1. Add to `config/constants.py` (constants/enums)
2. Add to `config/settings.py` (settings)
3. Use anywhere via `Settings.SOMETHING`

## File Size Guidelines

- **Keep files focused**: Each file should have one clear purpose
- **Component files**: 100-300 lines ideal
- **Service files**: 200-400 lines max
- **Split when**: File >500 lines or has >1 responsibility

## Code Quality Checklist

- [ ] Full type hints
- [ ] Docstrings on all public functions
- [ ] Custom exceptions for errors
- [ ] Logging for important operations
- [ ] No business logic in UI
- [ ] No UI code in core
- [ ] No direct session state access outside StateManager
- [ ] Clear separation of concerns
- [ ] Single responsibility per module

## Running the App

```bash
# Development
streamlit run app/main.py

# Production (with specific port)
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

## Why This Architecture?

### ✅ Advantages

1. **Maintainable**: Clear structure, easy to find code
2. **Testable**: Core logic testable without UI
3. **Scalable**: Easy to add features without breaking existing code
4. **Readable**: Each file has single purpose
5. **Professional**: Industry-standard patterns
6. **Type-Safe**: Catch errors at dev time, not runtime
7. **Debuggable**: Clear boundaries make debugging easier

### 🎯 When to Use This

- Production applications
- Team projects
- Long-term maintenance
- Apps >500 lines
- Apps needing testing
- Apps with complex business logic

### 🚫 When NOT to Use This

- Quick prototypes (<200 lines)
- One-off scripts
- Personal experiments
- Throwaway code

## Comparison to Previous Versions

### app_agent.py (Old Modular)
- ❌ Mixed concerns (UI + logic)
- ❌ Direct session state access
- ❌ No type safety
- ❌ Hard to test
- ✅ Modular

### app_agent_v2.py (Single File)
- ❌ Everything in one file
- ❌ Mixed concerns
- ❌ No separation of layers
- ❌ Hard to test
- ✅ Easy to read (for small apps)

### app/ (New Production)
- ✅ Clear separation of concerns
- ✅ Framework-agnostic core
- ✅ Full type safety
- ✅ Easy to test
- ✅ Modular and scalable
- ✅ Professional patterns

## Summary

This architecture is **production-grade** because it:

1. **Separates concerns** properly (UI, logic, state, config)
2. **Inverts dependencies** (core doesn't know about UI)
3. **Uses types** for safety
4. **Handles errors** gracefully
5. **Enables testing** at all levels
6. **Scales easily** as app grows
7. **Follows standards** (clean architecture, SOLID)

It's more code upfront, but **saves time long-term** through:
- Easier debugging
- Faster feature additions
- Safer refactoring
- Better team collaboration
- Higher code quality
