# Spec 092 — Google Photos Integration (Import + Export)

- **Status:** Draft
- **Created:** 2026-06-26
- **Decision report:** [`DECISION_REPORT.html`](./DECISION_REPORT.html)
- **Related:** spec-022 (geo/trip detection — consumes GPS), spec-089 (import run), spec-091 (export crops)

## 1. Problem

Users keep their photos in Google Photos. Today the pipeline only ingests local
folders (`source_directory` → `discover_images`) and only exports to folder/ZIP.
We must let users **import** photos/albums from Google Photos and **export** a
refined album back to Google Photos.

## 2. Hard constraint (verified vs. 2025/26 Google docs)

- **March 2025**: Google permanently removed library-read scopes. The only live-API
  way to read a user's existing photos is the **Picker API** (user hand-picks per session).
- **API downloads strip GPS** by design (`=d` keeps all EXIF *except* location).
  This conflicts with spec-022, which needs GPS.
- Export still works via `photoslibrary.appendonly` + `batchCreate`, but **only into
  app-created albums** (cannot touch the user's existing albums). Upload scopes are
  **restricted** (CASA audit) for public distribution.

## 3. Decisions (locked by product)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Import | **Committed.** Hybrid: Picker API (UX) + Takeout folder (GPS). |
| D2 | Export | **Committed.** Push selected album to an app-created Google Photos album. |
| D3 | GPS | **Accepted risk.** Ship without guaranteed GPS; **notify users** when location is absent and trip detection may be degraded. |
| D4 | Distribution | **Open** — default Testing mode (≤100 users, no audit). Public + CASA is a later business decision; does not block the build. |

## 4. GPS recovery strategy

Resolver runs a fallback chain per photo and **records the source**:

1. **Embedded EXIF** (present for Takeout / local; absent for Picker)
2. **Takeout sidecar** `supplemental-metadata.json` → `geoData`/`geoDataExif` (exact)
3. **Local-original match** (filename+timestamp, pHash fallback) → lift EXIF (exact)
4. **none** (recorded; surfaced to the user)

Gotchas (verified): sidecars renamed to `*.supplemental-metadata.json` and clipped at
~46 chars → fuzzy match; `0.0,0.0` = missing; prefer `geoData` over `geoDataExif`.
Maps-Timeline interpolation is **out of scope** for v1 (Google moved Timeline on-device
2024, export largely unavailable / ≤90-day retention).

## 5. Functional requirements

- **FR-001** User can import a Google Photos selection via an in-app Picker (browser OAuth).
- **FR-002** User can import a Google Takeout folder (bulk, GPS-preserving).
- **FR-003** Imported media materialize into a local cache dir consumed unchanged by `discover_images`.
- **FR-004** GPS resolver attaches `lat/lon/timestamp` per photo with a `source` tag; degrades gracefully to `none`.
- **FR-005** After a Picker import, UI reports GPS coverage and warns trip detection may be degraded, offering the Takeout path.
- **FR-006** User can export `selected_images` to a new app-created Google Photos album.
- **FR-007** Export batches uploads (≤50/call), retries on 429, and is idempotent on re-run.
- **FR-008** OAuth refresh token is persisted (OS keyring); silent refresh; clear re-consent UX.
- **FR-009** All Google integration is isolated behind a `gphotos/` package (single point of API change).

## 6. Architecture

Two edge adapters; existing 20+ pipeline steps untouched (images are path-by-reference).

```
SOURCE: auth → picker|takeout → download → cache/ ─┐
                         gps_resolver (exif→sidecar→local→none)
cache/ ─▶ discover_images ─▶ extract_geo_metadata ─▶ … ─▶ select_best ─▶ selected_images ─┐
SINK:                                                        upload + batchCreate ◀────────┘
```

New package `gphotos/` (`auth, picker, takeout, local_match, gps_resolver, cache,
ingest_source`); new sink `sim_bench/album/export/gphotos.py` beside `folder.py`/`zip.py`.
Follows spec-053: thin adapters, domain logic with `calc()` + typed `Inputs`/`Result`.

## 7. Testing

- **Unit (CI, no network):** `ut_TakeoutParser`, `ut_LocalMatch`, `ut_GpsResolver`,
  `ut_PickerSession` (mock HTTP), `ut_GphotosCache`, `ut_GphotosAuth`, `ut_GphotosExport`.
  Fixtures: a small synthetic Takeout folder.
- **Standalone live smoke (manual):** `scripts/gphotos_e2e_smoke.py` — opens a browser,
  imports via Picker, attaches GPS (Takeout/local), prints coverage. **Independent of
  Albumify** (no API/Streamlit/`sim_bench.db`). Plus `gphotos_auth_check.py`,
  `gphotos_picker_smoke.py`, `gphotos_export_smoke.py`.
- **E2E:** Takeout import → `discover_images` → `extract_geo_metadata` yields non-null GPS
  (the spec-022 unblock proof). Playwright: import button reaches a real run.
- **Baseline gate:** `pytest -m budapest` stays green (15 clusters / 340 faces).

## 8. Non-goals (v1)

- Maps-Timeline GPS interpolation.
- Writing into the user's pre-existing Google albums (Google disallows).
- Background/auto library sync (Picker is per-session by design).
- Public distribution / CASA (separate decision).

## 9. Open question

- **D4 distribution**: Testing (default) vs Public. Only affects Google's external review.

## 10. Done criteria

All FRs met; unit suite green in CI; `gphotos_e2e_smoke.py` shows import + GPS attach;
Budapest baseline green; `/code-review` → `REVIEW.md` with no unresolved High findings.
