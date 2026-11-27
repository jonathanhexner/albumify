# Tasks: Accounts and a public API (120)

**Spec:** [spec.md](spec.html) · [updates.md](updates.html)

## Design Notes
- **D1** A second FastAPI app (`sim_bench/public_api/`), deployed as a Modal web endpoint. The old API is not reachable from the internet and gets no new features.
- **D2** Sign-in by Clerk; the API only verifies its JWT against Clerk's published keys (JWKS). No passwords stored here.
- **D3** Ownership lives on the job row (`Job.owner_id`); every public read joins through it. Run rows keep `run_id == job_id`.
- **D4** Queue = the `jobs` table (status `queued`); `JobDispatcher.tick()` runs every 30 s on Modal and starts jobs in creation order.
- **D5** Uploads: signed PUT URLs per file into R2 `uploads/<job_id>/`; `UploadStager` writes 768 px inputs (EXIF kept, the existing `staging.downsample`) and the manifest; originals are swept after 7 days.
- **D6** Limits live in `configs/public.yaml` (1,000 photos, 25 MB, 6 GB, 3 albums / 3,000 photos a month, $50 a month, alert at 80 %, 7 days, max 2 running).

## W1: Accounts and sign-in
- [x] T1.1 `Account`, `Invite`, `ContentReport` models in `sim_bench/control/models.py`; migration for existing databases (SQLite + Neon)
- [x] T1.2 `AuthVerifier` in `sim_bench/public_api/auth.py` (PyJWT, JWKS cache, issuer and expiry checks)
- [x] T1.3 `AccountRepo.get_or_create` (invite check), `record_consent`
- [x] T1.4 Tests `tests/public_api/test_auth.py`, `tests/public_api/test_accounts.py` (locally signed test keys, no network) — R1 R2 R3

**Checkpoint:** a forged token gets 401; an uninvited email gets 403; no consent → no upload.

## W2: Ownership and fast reads
- [x] T2.1 `Job.owner_id` (indexed) + migration; owner-less jobs stay the owner's (`owner_id` NULL = the owner's own runs)
- [x] T2.2 `PublicAlbumRepo` in `sim_bench/public_api/album_repo.py`: list, album, photos (best 30 / all), people, rename; each one or two queries
- [x] T2.3 Tests `tests/public_api/test_isolation.py` (two users, every read), `tests/public_api/test_query_counts.py`; add to the `needs_neon` parity run — R4 R11

**Checkpoint:** user B gets 404 on everything of user A; query counts ≤ 2; real Neon timing < 1 s on Slovakia26.

## W3: Uploads, limits, quota, spend cap
- [x] T3.1 `R2ObjectStore.presigned_put`, `presigned_get` in `sim_bench/storage/object_store.py`
- [x] T3.2 `Limits`, `QuotaGuard`, `SpendGuard` in `sim_bench/public_api/limits.py`; `configs/public.yaml`
- [x] T3.3 `UploadService.create` (checks, estimate, runner by size) and `complete` (objects present → queued job)
- [x] T3.4 Tests `tests/public_api/test_uploads.py`, `tests/public_api/test_limits.py`; real R2 signed PUT — R5 R6 R7 R8

**Checkpoint:** 1,001 photos, a 30 MB file or a `.exe` are refused before anything is stored; the 4th album of the month gets 429.

## W4: Job queue and staging on Modal
- [ ] T4.1 `SubmitInputs.owner_id`; `JobService.create` from an upload (no local folder)
- [ ] T4.2 `JobDispatcher` in `sim_bench/serving/dispatcher.py`
- [ ] T4.3 `UploadStager` in `sim_bench/serving/upload_stager.py`
- [ ] T4.4 Modal functions `stage_upload`, `dispatch` (every 30 s) in `sim_bench/serving/modal_app.py`
- [ ] T4.5 Tests `tests/serving/test_dispatcher.py`, `tests/serving/test_upload_stager.py`; real Modal run of 50 uploaded photos — R9 R10

**Checkpoint:** three queued jobs start in order, two at a time; killing the API mid-run loses nothing.

## W5: Album endpoints
- [ ] T5.1 Routes in `sim_bench/public_api/app.py`, response models in `sim_bench/public_api/schemas.py`
- [ ] T5.2 Rename person; delete album (`PipelineService.delete_run` + R2 prefix); delete account (`AccountRepo.delete_everything`)
- [ ] T5.3 Tests `tests/public_api/test_people.py`, `tests/public_api/test_delete.py`; real R2 listing empty after delete — R12 R13

## W6: Retention, reports, deploy
- [ ] T6.1 `RetentionSweeper` in `sim_bench/serving/retention.py`; Modal `sweep` daily
- [ ] T6.2 `POST /v1/reports` → `ContentReport`
- [ ] T6.3 Modal `public_api` web endpoint, secrets from `albumify-cloud`, `/health`
- [ ] T6.4 Tests `tests/serving/test_retention.py`, `tests/public_api/test_reports.py`; real deploy + secret scan — R14 R15

## Close-out
- [ ] T7.1 One invited account: 200 photos from upload to album on the deployed API (real data)
- [ ] T7.2 Full suite, `needs_neon` parity, Budapest gate, `/code-review` → REVIEW.md
