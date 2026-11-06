# spec-117 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Draft 2026-11-06

## 2026-11-06 — Drafted

Owner, after the L4 path was explained ([diagram](https://claude.ai/artifact/K3WQv7SkBsConZVnJZhh9f)): "from my perspective,
L4 should run and write results to DB. Also, DB is used for polling status and telemetry. If we lost connection we just
query DB for status and results." Then: "go with A" and "move everything to Neon".

Accounts were set up and checked the same day:

| Claim | Evidence |
|---|---|
| Neon reachable with the `.env` URL | Scratch check: PostgreSQL 18.6, pooled host, temp table write/read: PASS |
| R2 bucket usable with the scoped token | Scratch check: put, read back (match), delete in `albumify`: PASS |
| Modal secret exists | `modal secret list`: `albumify-cloud`, created 2026-11-06 11:20 |
| Leaked password not in use | `.env` password differs from the one pasted in chat |

Code map (read-only) found: nothing in the API loads `.env`; the API DB URL is always SQLite; `control.db` already
supports Postgres; two tables are both named `people`; status reaches the app through the job row, a local
`progress.json` and an in-memory dict; no Postgres driver or S3 client is installed.
