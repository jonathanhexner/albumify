# Spec 092 — Tasks

Legend: `[ ]` open · `[>]` in progress · `[x]` done. Each phase ends with a **runnable gate**.

---

## P0 — Scaffold
- [ ] T0.1 Create Google Cloud project; OAuth client (Desktop / installed-app); enable Photos Picker API. **(user)**
- [ ] T0.2 Add scope `photospicker.mediaitems.readonly`; add self as test user; download `client_secret.json`. **(user)**
- [x] T0.3 Create `gphotos/` package; add deps `google-auth`, `google-auth-oauthlib`, `requests`; `pip install -e .`.
- [ ] T0.4 Confirm D4 distribution scope (Testing default) with product.
- **GATE:** consent screen reachable for the test user.

## P1 — GPS core (first value · unblocks spec-022 · CI-testable)
- [ ] T1.1 `gphotos/takeout.py` — walk export, fuzzy-match `*.supplemental-metadata.json` (handle 46-char truncation), parse `geoData`/`geoDataExif`; treat `0.0,0.0` as missing; prefer `geoData`.
- [ ] T1.2 `gphotos/local_match.py` — match cache file → local original by filename+timestamp, pHash fallback; lift EXIF GPS.
- [ ] T1.3 `gphotos/gps_resolver.py` — `calc()` fallback chain exif→sidecar→local→none; typed `Result` with `lat/lon/timestamp/source`.
- [ ] T1.4 Tests `ut_TakeoutParser`, `ut_LocalMatch`, `ut_GpsResolver` + synthetic Takeout fixture (no network).
- **GATE:** `.venv/Scripts/python -m pytest tests/gphotos -k "takeout or resolver or local"` green.

## P2 — Auth
- [x] T2.1 `gphotos/auth.py` — `InstalledAppFlow.run_local_server()` loopback; persist refresh token (JSON file under `~/.sim_bench/`; keyring deferred to P7); silent refresh.
- [x] T2.2 `scripts/gphotos_auth_check.py` — opens browser, authenticates, prints token validity + granted scopes.
- [ ] T2.3 `ut_GphotosAuth` with a fake credential store. *(deferred — covered by live auth_check for now)*
- **GATE:** `.venv/Scripts/python scripts/gphotos_auth_check.py` → "token OK". **[PASSED 2026-06-28 — valid token, refresh token present]**

## P3 — Picker import
- [x] T3.1 `gphotos/picker.py` — `sessions.create`, poll `sessions.get` (respect interval/timeout), `mediaItems.list` (paginated), download `=d` bytes (bearer via `AuthorizedSession`).
- [x] T3.2 `gphotos/cache.py` — materialize to cache dir; dedupe; idempotent re-download (manifest).
- [x] T3.3 Tests `ut_PickerSession` + `ut_GphotosCache` (mock HTTP, no network) — 11 passing.
- [x] T3.4 `scripts/gphotos_picker_smoke.py` — opens browser, user picks, downloads N to cache, prints count.
- **GATE:** `scripts/gphotos_picker_smoke.py` downloads ≥1 file. **[PASSED 2026-06-28 — 2 full-res photos downloaded; EXIF confirmed: timestamp+camera survive, GPS stripped as expected]**

## P4 — E2E smoke (the requested standalone test)
- [x] T4.1 `gphotos/ingest_source.py` — `import_from_google_photos()` adapter: auth → pick → download → returns `IngestResult(source_directory, items)` for `discover_images`. Injectable auth/client; 3 unit tests.
- [x] T4.2 `scripts/gphotos_e2e_smoke.py` — imports then runs `execute_spec` over the cache dir (`--steps`, `--full`, `--out`). No Albumify/API/Streamlit/DB.
- **GATE:** import → `discover_images` + `extract_geo_metadata` runs clean. **[PASSED 2026-06-28 — 2 imported photos flow through the real pipeline; success=True, 2 images, timestamps survive, GPS absent as expected]**
- *(GPS resolver / Takeout flags deferred with the P1 GPS track, set aside per product decision.)*

## P5 — UI wiring
- [x] T5.1 FC `app/face_clustering/tabs/run_tab.py` — "Import from Google Photos" button → sets `last_image_dir`.
- [x] T5.2 Albumify `app/streamlit/components/album_selector.py` — same button in the album creator (sets `new_album_source`). *(creator lives on the Albums page, not configure.py.)*
- [x] T5.3 GPS notice (FR-005): widget shows "imports carry no GPS; trip/geo detection limited; use Takeout for location" after each import.
- [x] T5.4 Playwright: button visible in FC run tab (screenshot verified, client-secret gate passes).
- **Shared widget:** NEW `gphotos/ui_streamlit.py` `render_import_button(key, target_key)` — lazy streamlit import, keeps core package framework-agnostic; used by both apps.
- **GATE:** button renders in a live app. **[PASSED 2026-06-28 — FC run tab screenshot shows the button above Image directory; app boots clean.]**
- *(Albumify side wired + compiles + boots; full click-through needs the API server running.)*

## P6 — Export
- [x] T6.1 `gphotos/uploader.py` (`LibraryClient`: upload_bytes/create_album/batch_create ≤50, 429+5xx backoff) + `gphotos/export_album.py` (`export_album_to_google_photos()` -> `ExportResult`; manifest-based idempotency: reuse album by title, skip already-uploaded). *(Placed in `gphotos/` not `sim_bench/album/export/` to keep all Google API behind one boundary.)*
- [x] T6.2 "Export to Google Photos" expander in `app/streamlit/components/export_panel.py` (pushes `get_selected_images(job_id)`).
- [x] T6.3 `ut_GphotosUploader` + `ut_GphotosExport` (8 tests, mock HTTP) + `scripts/gphotos_export_smoke.py`.
- [x] Auth: added `APPEND_SCOPE`/`EDIT_SCOPE` + separate `LIBRARY_TOKEN_PATH` (export consent is independent of the picker token).
- **GATE:** `scripts/gphotos_export_smoke.py` creates album, verifies item count. **[CODE READY — live run needs user console step: enable Library API + add appendonly scope + re-consent.]**

## P7 — Harden + close out
- [ ] T7.1 Quota/backoff hardening; clear re-consent UX; redact tokens from logs.
- [ ] T7.2 Docs: update `docs/architecture/` (data_flow + classes) for the new adapters; CHANGES_LOG entries.
- [ ] T7.3 `.venv/Scripts/python -m pytest -m budapest tests/face_clustering/e2e_budapest/ -v` green (15/340).
- [ ] T7.4 `/code-review` → `REVIEW.md`; resolve High findings.
- **GATE:** baseline green + REVIEW.md clean → flip spec to Implemented.
