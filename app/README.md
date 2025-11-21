# Photo Organization App

A production-grade Streamlit application for organizing photos using AI agents.

## Quick Start

```bash
# Run the app
streamlit run app/main.py

# Or from project root
streamlit run app/main.py
```

Then:
1. Click "Initialize Agent" in the sidebar
2. Enter your photo directory path
3. Start asking questions about your photos!

## Architecture

This app follows **clean architecture** principles with clear separation of concerns:

```
app/
├── config/      # Configuration and constants
├── core/        # Business logic (framework-agnostic)
├── state/       # State management
├── ui/          # Presentation layer (Streamlit)
└── main.py      # Entry point
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Features

- 📁 **Event Clustering** - Automatically group photos by event
- 👤 **Face Recognition** - Organize photos by people
- 🗺️ **Landmark Detection** - Sort travel photos by location
- ⭐ **Quality Assessment** - Find your best photos
- 🏷️ **Smart Tagging** - Automatic scene and object detection

## Example Queries

- "Organize my photos by event"
- "Find my best 10 photos"
- "Group photos by person"
- "Show me all portrait photos"
- "Find photos from my vacation"
- "Organize travel photos by landmarks"

## Key Design Principles

1. **Separation of Concerns**: UI, business logic, state, and config are separate
2. **Type Safety**: Full type hints throughout
3. **Testability**: Core logic can be tested without UI
4. **Modularity**: Easy to extend and modify
5. **Professional**: Industry-standard patterns (Service Layer, Repository, DI)

## Project Structure

### Configuration (`config/`)
- `constants.py` - Enums, constants, example queries
- `settings.py` - Application settings and agent configurations

### Core (`core/`)
Framework-agnostic business logic:
- `models.py` - Domain models (AppState, ImageLibrary, ChatMessage, etc.)
- `services.py` - Services (AgentService, ImageService, ConversationService)
- `exceptions.py` - Custom exceptions

### State (`state/`)
- `manager.py` - StateManager (bridges Streamlit session state with domain models)

### UI (`ui/`)
Streamlit-specific presentation:
- `pages.py` - Page orchestration
- `styles.py` - CSS styling
- `components/` - Reusable components (sidebar, chat, status)

## Development

### Running Tests

```bash
# Unit tests (core layer)
pytest app/core/

# Integration tests (requires Streamlit test framework)
pytest app/ui/
```

### Code Quality

- All code has type hints
- All public functions have docstrings
- Business logic is in services, not UI
- State access goes through StateManager only

### Adding Features

1. Add domain model in `core/models.py` (if needed)
2. Add business logic in `core/services.py`
3. Add state methods in `state/manager.py` (if needed)
4. Add UI component in `ui/components/`
5. Wire together in `ui/pages.py`

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed guidelines.

## Error Handling

The app uses custom exceptions for clear error handling:
- `AgentError` - Agent operations failed
- `ImageLoadError` - Image loading failed
- `ValidationError` - Input validation failed

All errors are:
- Logged with full details
- Displayed to user with friendly messages
- Never silently swallowed

## Dependencies

Core dependencies:
- `streamlit` - Web framework
- `sim_bench` - Image analysis package

See parent `requirements.txt` for full list.

## Deployment

### Local

```bash
streamlit run app/main.py
```

### Production

```bash
# With specific port and address
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0

# With config file
streamlit run app/main.py --server.enableCORS false
```

## Troubleshooting

### Agent not initializing
- Check logs for detailed error
- Ensure sim_bench is installed correctly
- Verify agent type is available

### No images found
- Check directory path is correct
- Ensure directory contains .jpg/.jpeg/.png files
- Check file permissions

### App crashes on startup
- Check Python version (3.10+ required)
- Ensure all dependencies are installed
- Check logs in console

## License

MIT License - see parent LICENSE file for details.

## Contributing

This app follows strict architectural guidelines. Please:
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) first
2. Follow separation of concerns
3. Add type hints to all code
4. Write tests for business logic
5. Keep components focused and small

## Version History

- **2.0.0** - Production architecture rewrite (clean architecture, full type safety)
- **1.0.0** - Initial modular version
