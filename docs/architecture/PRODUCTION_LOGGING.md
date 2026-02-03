# Production Logging Guide

This document explains the production logging architecture for sim-bench, designed for someone familiar with Python's basic `logging` module but new to production-grade logging.

---

## Table of Contents

1. [Why Production Logging is Different](#why-production-logging-is-different)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Details](#implementation-details)
5. [Configuration Reference](#configuration-reference)
6. [How to Use It](#how-to-use-it)

---

## Why Production Logging is Different

### Basic Python Logging (What You Know)

```python
import logging
logger = logging.getLogger(__name__)
logger.info("User logged in")
logger.error("Something went wrong", exc_info=True)
```

This works fine for development, but in production you face challenges:

| Challenge | Why It Matters |
|-----------|----------------|
| **Multiple servers** | Logs are scattered across machines |
| **High volume** | Can't read through thousands of log lines |
| **Debugging distributed systems** | One user action touches multiple services |
| **Log management** | Files grow forever without rotation |
| **Alerting** | Need to detect errors automatically |

### Production Logging Adds

1. **Structured logs (JSON)** - Machines can parse and search
2. **Correlation IDs** - Track a single request across services
3. **Log rotation** - Prevent disk from filling up
4. **Centralized collection** - All logs in one searchable place
5. **Error reporting** - Catch and surface problems automatically

---

## Key Concepts Explained

### 1. Structured Logging (JSON Format)

**Basic logging:**
```
2024-01-30 10:30:00 - myapp - INFO - User created album: vacation photos
```

**Structured logging:**
```json
{
  "timestamp": "2024-01-30T10:30:00.000Z",
  "level": "INFO",
  "logger": "myapp",
  "message": "User created album",
  "album_name": "vacation photos",
  "user_id": "user123",
  "duration_ms": 45
}
```

**Why JSON?**
- Log aggregation tools (CloudWatch, Datadog, ELK) can parse it
- You can search: "show me all errors where `album_name` contains 'vacation'"
- You can create dashboards: "average request duration per endpoint"
- You can alert: "notify me when `error.type` = 'DatabaseError'"

### 2. Correlation IDs

**The Problem:**
```
Server A log: "Received request for album list"
Server B log: "Database query executed"
Server A log: "Error: timeout"
Server B log: "Query completed successfully"
```

Which log lines belong to the same request? You can't tell.

**The Solution - Correlation ID:**
```
Server A: {"message": "Received request", "correlation_id": "abc-123"}
Server A: {"message": "Calling Server B", "correlation_id": "abc-123"}
Server B: {"message": "Database query", "correlation_id": "abc-123"}
Server A: {"message": "Error: timeout", "correlation_id": "abc-123"}
```

Now you can filter: "show me all logs where `correlation_id` = 'abc-123'" and see the complete request flow.

**How It Works:**
1. First service generates a unique ID (UUID)
2. ID is passed in HTTP header: `X-Correlation-ID: abc-123`
3. Each service logs with this ID
4. Response includes the ID so users can report it

### 3. Log Rotation

**The Problem:**
Without rotation, your `api.log` grows forever until disk is full.

**Size-Based Rotation:**
```
logs/
  api.log        # Current file (writing here)
  api.log.1      # Previous file (100MB)
  api.log.2      # Older file (100MB)
  api.log.3      # Oldest file (deleted when api.log.4 would be created)
```

When `api.log` reaches 100MB:
1. `api.log.2` → `api.log.3` (oldest deleted if over limit)
2. `api.log.1` → `api.log.2`
3. `api.log` → `api.log.1`
4. New empty `api.log` created

**Time-Based Rotation (Alternative):**
- Rotate at midnight every day
- Keep 30 days of logs
- Delete files older than 30 days

### 4. Request/Response Middleware

**What is Middleware?**
Code that runs for EVERY request, before and after your route handler.

```
Request comes in
    ↓
[Middleware: Start timer, generate correlation ID]
    ↓
Your route handler runs
    ↓
[Middleware: Calculate duration, log request details]
    ↓
Response sent
```

**What We Log:**
```json
{
  "message": "GET /api/v1/albums - 200 (45ms)",
  "request": {
    "method": "GET",
    "path": "/api/v1/albums",
    "status_code": 200,
    "duration_ms": 45,
    "client_ip": "192.168.1.100"
  }
}
```

**What We DON'T Log (for security/performance):**
- Request bodies (might contain passwords)
- Response bodies (large, might contain sensitive data)
- Health check endpoints (too noisy)

### 5. Exception Handlers

**Without Exception Handlers:**
```python
@app.get("/albums/{id}")
def get_album(id: str):
    album = db.get(id)  # Raises exception if not found
    return album        # Never reached
# User sees: 500 Internal Server Error (no details)
# Logs: Nothing useful
```

**With Exception Handlers:**
```python
@app.exception_handler(Exception)
def handle_exception(request, exc):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal error", "correlation_id": "abc-123"}
    )
# User sees: {"error": "Internal error", "correlation_id": "abc-123"}
# Logs: Full stack trace with correlation ID
```

User can report: "I got an error, correlation_id was abc-123"
You can search logs for that ID and see exactly what happened.

### 6. Context Variables (contextvars)

**The Problem:**
How do you pass the correlation ID through your entire codebase without changing every function signature?

```python
# Ugly approach - passing through every function
def get_album(id, correlation_id):
    return db.query(id, correlation_id)

def db.query(id, correlation_id):
    logger.info("Query", extra={"correlation_id": correlation_id})
```

**The Solution - Context Variables:**
```python
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar("correlation_id")

# Set once at the start of the request
correlation_id.set("abc-123")

# Access anywhere without passing it
def get_album(id):
    return db.query(id)

def db.query(id):
    logger.info("Query")  # Formatter automatically adds correlation_id
```

Context variables are:
- **Thread-safe**: Each request has its own value
- **Async-safe**: Works with `async/await`
- **Transparent**: Functions don't need to know about it

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ NiceGUI Frontend (localhost:8080)                               │
│                                                                 │
│   error_handler.py ──→ Reports errors to backend                │
│   api_client.py ──→ Captures X-Correlation-ID from responses    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP + WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Backend (localhost:8000)                                │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ RequestLoggingMiddleware                                │   │
│   │  • Generate/extract correlation ID                      │   │
│   │  • Start timer                                          │   │
│   │  • Log request when complete                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Exception Handlers                                      │   │
│   │  • Catch all exceptions                                 │   │
│   │  • Log with stack trace                                 │   │
│   │  • Return correlation ID in error response              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Routers & Services                                      │   │
│   │  • Use get_logger(__name__)                             │   │
│   │  • Correlation ID automatically included                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ JSON Formatter                                          │   │
│   │  • Converts log records to JSON                         │   │
│   │  • Adds correlation ID, timestamp, service name         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                             ↓                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ RotatingFileHandler          ConsoleHandler             │   │
│   │  • logs/api.log              • stdout (for Docker)      │   │
│   │  • Rotates at 100MB          • Same JSON format         │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### File Structure

```
sim_bench/api/logging/
    __init__.py       # Public exports
    config.py         # Environment variable configuration
    context.py        # Correlation ID management (contextvars)
    formatter.py      # JSON log formatter
    middleware.py     # Request/response logging
    handlers.py       # Exception handlers
    logger.py         # Logger factory with rotation

sim_bench/api/routers/
    errors.py         # Client error reporting endpoint (NEW)

app/nicegui/
    error_handler.py  # Frontend error capture (NEW)
    api_client.py     # Updated with correlation ID tracking
```

### How the Pieces Connect

**1. Application Startup (`main.py`):**
```python
from sim_bench.api.logging import setup_logging, RequestLoggingMiddleware, setup_exception_handlers

# Initialize logging first
setup_logging()

app = FastAPI()

# Add middleware (order matters - logging should be outer)
app.add_middleware(RequestLoggingMiddleware)

# Register exception handlers
setup_exception_handlers(app)
```

**2. Request Flow:**
```
1. Request arrives
2. RequestLoggingMiddleware:
   - Checks for X-Correlation-ID header (or generates one)
   - Stores in context variable
   - Starts timer
3. Your route handler runs
   - logger.info("...") automatically includes correlation_id
4. If exception:
   - Exception handler catches it
   - Logs full stack trace with correlation_id
   - Returns error response with correlation_id
5. RequestLoggingMiddleware:
   - Calculates duration
   - Logs: "GET /api/v1/albums - 200 (45ms)"
   - Adds X-Correlation-ID to response header
```

**3. Frontend Error Reporting:**
```
1. JavaScript/Python error occurs in NiceGUI
2. error_handler.py catches it
3. Sends POST to /api/v1/errors/report with:
   - error type, message, stack trace
   - URL where it happened
4. Backend logs it with correlation_id
5. All errors (frontend + backend) in one log file
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | `json` | Output format: `json` or `text` |
| `LOG_FILE_PATH` | `logs/api.log` | Where to write log files |
| `LOG_MAX_SIZE_MB` | `100` | Rotate when file reaches this size |
| `LOG_BACKUP_COUNT` | `5` | Number of rotated files to keep |
| `LOG_INCLUDE_CONSOLE` | `true` | Also output to stdout |
| `ENVIRONMENT` | `development` | Included in every log entry |

### Example `.env` File

```bash
# Development (verbose, text format)
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_INCLUDE_CONSOLE=true
ENVIRONMENT=development

# Production (structured, minimal console)
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_INCLUDE_CONSOLE=false
ENVIRONMENT=production
```

---

## How to Use It

### In Your Code

```python
from sim_bench.api.logging import get_logger

logger = get_logger(__name__)

def process_album(album_id: str):
    logger.info(f"Processing album {album_id}")

    try:
        result = do_heavy_computation()
        logger.info(f"Processing complete", extra={"result_count": len(result)})
        return result
    except ValueError as e:
        logger.warning(f"Invalid album data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing album", exc_info=True)
        raise
```

### Log Levels Guide

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed diagnostic info (usually disabled in prod) | Variable values, loop iterations |
| `INFO` | Normal operations worth recording | Request received, job started/completed |
| `WARNING` | Something unexpected but handled | Retry attempted, fallback used |
| `ERROR` | Something failed | Exception caught, operation failed |
| `CRITICAL` | System is unusable | Database connection lost, out of memory |

### Searching Logs

**In CloudWatch/Datadog:**
```
# Find all errors for a specific correlation ID
correlation_id="abc-123" AND level="ERROR"

# Find slow requests (over 1 second)
request.duration_ms > 1000

# Find all errors in the last hour
level="ERROR" AND timestamp > now() - 1h
```

**Locally with jq:**
```bash
# Pretty print
cat logs/api.log | jq .

# Filter errors
cat logs/api.log | jq 'select(.level == "ERROR")'

# Find by correlation ID
cat logs/api.log | jq 'select(.correlation_id == "abc-123")'
```

---

## Summary

| Component | What It Does |
|-----------|--------------|
| **JSON Formatter** | Makes logs machine-readable |
| **Correlation ID** | Links related log entries across services |
| **Middleware** | Automatically logs every request |
| **Exception Handlers** | Catches errors, logs stack traces |
| **Rotation** | Prevents disk from filling up |
| **Client Error Endpoint** | Frontend reports errors to backend |

The goal: When something breaks in production, you can find the correlation ID and see exactly what happened, without adding logging code to every function.
