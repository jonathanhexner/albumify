# spec-068 — v2 tab render telemetry (driver-agnostic test/debug signal)

**Created**: 2026-06-03
**Status**: Code Review (impl complete 2026-06-03; tests green; /code-review pending)
**Priority**: P2 (testability; unblocks faster debugging of v2 tab failures)
**Predecessors**: spec-063/064/065 (the tabs being instrumented), spec-067
  (surfaced the need — three slow browser runs spent guessing *which tab ran*)

---

## Problem

When a v2 tab renders blank in a test, we currently cannot tell **why** from
the outside:

- Did the render function even get called?
- Did it resolve a run dir, or bail at the "no run loaded" guard?
- Did it get data (N rows / N faces), or hit an empty result?
- Or did the browser just fail to paint something that *was* produced?

Today the only signals are (a) Playwright DOM scraping + failure screenshots
(slow, and a blank screenshot can mean any of the above) and (b) the headless
AppTest element list (good, but only available when running AppTest, not
during a real-browser run). During spec-067 this gap cost three multi-minute
browser runs to answer a question one log line would have answered instantly.

**Streamlit has no HTTP "response" to log** — it streams widgets to the
client over a websocket. The meaningful telemetry is *app-side state*: which
render function ran, the run dir it resolved, and the counts/values produced.

## What we build

A consistent **start / done** log pair in every v2 tab render function. One
helper to keep the format uniform; ~2 log calls per tab.

```python
# app/face_clustering_v2/_telemetry.py  (new, tiny)
import logging
logger = logging.getLogger("fc_app_v2.tabs")

def tab_start(name: str, run_dir) -> None:
    logger.info("tab.start name=%s run_dir=%s", name, run_dir)

def tab_done(name: str, **counts) -> None:
    # counts rendered as k=v, ASCII only (Windows CLI safe)
    kv = " ".join(f"{k}={v}" for k, v in counts.items())
    logger.info("tab.done name=%s %s", name, kv)

def tab_skipped(name: str, reason: str) -> None:
    logger.info("tab.skipped name=%s reason=%s", name, reason)
```

Each tab calls `tab_start` after resolving its run dir, `tab_skipped` on the
"no run loaded" / early-return branches, and `tab_done` with its key counts
just before returning. Examples of the counts per tab:

| Tab | `tab_done` counts |
|-----|-------------------|
| Run | (logs on submit only) |
| Cluster Analysis | `n_clusters`, `selected_cluster`, `n_faces` |
| Face Analysis | `face_id`, `cluster` |
| Merged Clusters | `n_rows` |
| Quality | `n_items`, `n_rejected`, `n_rows` |
| Recluster | `n_clusters` (on completion), `parent_run_id` |
| History | `n_runs` |

Level: `INFO` for the start/done/skipped milestones (cheap, one line each).
Reserve `DEBUG` for any verbose per-item detail if a tab wants it later.

## Why this is the right layer

- **Driver-agnostic**: identical output whether AppTest or a real browser
  drives the app. The log is written by the app, not the test.
- **Answers "which tab ran with what data"** — the exact question screenshots
  cannot answer.
- It does **NOT** replace the binding browser paint check (CLAUDE.md V2
  baseline gate). A log line "n_items=5" is not proof pixels appeared. The
  browser gate stays; this is the *fast* signal for everything else.

## Acceptance criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| AC1 | `_telemetry.py` helper exists with `tab_start`/`tab_done`/`tab_skipped` | unit test |
| AC2 | All 7 v2 tabs call `tab_start` + (`tab_done` or `tab_skipped`) on every return path | grep + code review |
| AC3 | Running main.py via AppTest with a seeded run emits `tab.done name=quality ...` (and the other tabs) — assertable via `caplog` | new test `test_v2_tab_telemetry.py` |
| AC4 | "No run loaded" path emits `tab.skipped reason=no_run_loaded` instead of `tab.done` | same test, no-seed case |
| AC5 | Output is ASCII-only, single line per event (Windows CLI safe) | code review + test |
| AC6 | No behaviour change to what renders — telemetry is logging only | existing v2 AppTest smoke stays green |

## Out of scope

- Logging actual rendered pixel/DOM content (that's the browser gate's job).
- Structured/JSON logging or a metrics backend — plain key=value INFO lines.
- Instrumenting the legacy `app/face_clustering/` tabs.

## Effort estimate

~1.5 h: tiny helper + 2 calls × 7 tabs + one `caplog` test.
