# spec-120 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress 2026-11-22 (owner approved; W1-W3 done)

## 2026-11-28 — W3 done: uploads, limits, quota, spend cap

`POST /v1/uploads` checks consent, count, size, type, the monthly quota and the spend cap, then returns one signed R2
PUT URL per file, the expected wait and the runner. `POST /v1/uploads/{id}/complete` checks every file arrived at its
declared size, checks quota and spend again, and queues the job with its owner. No job and no quota until complete.
New `uploads` table; limits in `configs/public.yaml`. Plain-language explainer of what W4 adds: [why_w4.html](why_w4.html).

| Claim | Evidence |
|---|---|
| R5 limits before anything is stored: 1,001 photos, a 30 MB photo, > 6 GB, `.exe` / `.mp4` / `.gif` / a lying content type are refused, nothing written | `tests/public_api/test_uploads.py::ut_UploadLimits` (8 tests) + `ut_UploadRoutes::test_refusals_over_http` (400 / 413 / 415 / 422) |
| R5 on the real bucket: the signed PUT takes the declared size and type; bigger, smaller or another type get 403 | `tests/storage/test_object_store.py::test_real_r2_signed_put` (`-m needs_r2`): passed 2026-11-28 |
| R5 a client that cannot sign the size never hands out a URL | `ut_R2SignedUrls::test_a_client_that_cannot_sign_the_size_gives_no_url`: boto's default (older) signature leaves the size unsigned; `from_env` now forces SigV4 and `presigned_put` refuses otherwise |
| R6 >= 300 photos run on the GPU, fewer on Modal's CPU; wait = upload + processing | `test_uploads.py::ut_UploadEstimate::test_estimate_and_runner_by_size`. CPU figures (3 s/photo, $0.20/h) are estimates until the W4 real run |
| R7 the 4th album of the month gets 429 with the reset date; resets on the 1st (UTC); per account; unfinished uploads cost nothing; re-checked at complete | `tests/public_api/test_limits.py::ut_MonthlyQuota` (7 tests). The photo limit cannot bite at the defaults (3 x 1,000 = 3,000), so its one test lowers it |
| R8 passing $40 of $50 logs one alert; an upload that would pass $50 is refused (503); re-checked at complete; last month does not count | `test_limits.py::ut_SpendCap` (4 tests) |
| R3 no upload before face-grouping consent | `test_uploads.py::test_no_upload_without_face_consent`, `test_no_consent_is_403_and_no_token_is_401` |
| R1 the two new routes need a valid token | `tests/public_api/test_auth.py::test_every_other_route_needs_a_valid_token` (now 4 routes), `test_every_route_is_covered` |
| On Postgres too | `tests/storage/test_neon_parity.py::test_public_api_suite_passes_on_postgres`: passed in 15.7 min. **A first run failed** (concurrently with the local regression run); its error text was lost (my command kept only the last 15 lines), so the cause is not known. A third run is under way with the full output kept |
| Nothing else broke | `tests/public_api tests/api tests/control tests/serving tests/storage tests/architecture tests/streamlit`: 600 passed, 0 failed |

Found on the way: `PyJWT` (in `requirements.txt` since W1) was missing from `D:/sim-bench/.venv`; installed. W1-W2 had not
added the public tables to `docs/architecture/`; now in `db_schemas.html` and `classes.html`.

## 2026-11-22 — W2 done: ownership and fast reads

`Job.owner_id` (NULL = the owner's own runs, never visible on the public site); older databases gain the column and
its index on open (`_add_missing_indexes`, new). `PublicAlbumRepo`: list, album, photos (best 30 / all), people,
rename person; each filtered by owner and at most two queries.

| Claim | Evidence |
|---|---|
| R4 another user's album, photos, people or person is not found; the owner's own runs are nobody's | `tests/public_api/test_isolation.py::ut_Isolation` (two users, every read and the rename) |
| R4 on Postgres too | `tests/storage/test_neon_parity.py::test_public_api_suite_passes_on_postgres` (new, `tests/public_api` alone): passed on Neon in 7.7 min; the full parity run now includes `tests/public_api` |
| R11 query counts: list 1, album 1, photos 2, people 2, rename 2 | `tests/public_api/test_isolation.py::ut_QueryCounts` |
| R11 timing on real Neon, run `f7d8ba85` (715 photos, 659 scored, 7 people), median of 5 | list 0.31 s · album 0.30 s · best 30 0.46 s · all 659 photos 0.62 s (once 1.07 s) · people 0.46 s. Run owned by a probe account for the test, reset after |
| Nothing else broke | `tests/public_api tests/api tests/control tests/serving tests/architecture tests/streamlit`: 509 passed, 0 failed |

**Not yet (R11 ◐):** a 1,000-photo album is timed with the real 1,000-photo run (W6). The real Neon `jobs` table
now has the empty `owner_id` column and its index (added when the timing script opened it, as the app does at start).

## 2026-11-22 — W1 done: accounts and sign-in

Owner: "go ahead". New package `sim_bench/public_api/` (a second FastAPI app; the owner API is untouched).

| Claim | Evidence |
|---|---|
| R1 a forged, expired, other-issuer, other-key, HS256-confused or email-less token gets 401; `/health` needs none; every route is covered | `tests/public_api/test_auth.py`: 15 passed (tokens signed with a local RSA key standing in for Clerk's) |
| R2 only invited emails get an account; the same email under another sign-in is never merged | `tests/public_api/test_accounts.py`: `test_uninvited_email_is_waitlisted`, `test_an_email_taken_by_another_sign_in_is_not_merged`, 403 through the API |
| R3 consent is recorded once, through `POST /v1/me/consent` | `test_upload_needs_face_consent`, `test_consent_is_recorded_through_the_api` |
| Existing databases get the new tables | a copy of the real local `control.db`: `accounts`, `invites`, `content_reports` created on open, 20 jobs kept |
| Nothing else broke | `tests/public_api tests/control tests/architecture`: 239 passed, 0 failed |

**Not yet:** R1 against a real Clerk issuer (needs the Clerk account: owner action, see below); R3's "no upload
without consent" is enforced when uploads exist (W3). Clerk setup: create the application (Google + email link),
add `"email": "{{user.primary_email_address}}"` to the session-token template, put `CLERK_ISSUER` in the Modal secret.

## 2026-11-22 — drafted

Owner: "go with your recommendations, draft the spec" (productization plan, steps 3-4). Decisions taken from the
plan's recommendations: accounts in v1 with Clerk sign-in (Google + email link), invite list at launch, 3 albums and
3,000 photos per account per month, $50 a month spend cap with an 80 % alert, up to 1,000 photos per upload with a
wait estimate, originals kept 7 days and albums until deleted, face-grouping consent and delete-everything, US
region, private albums, rename-person as the only edit, API on Modal, terms + report button for content safety.

| Claim | Evidence |
|---|---|
| Spec format | `scripts/prd_lint.py specs/120-public-api-accounts/spec.md --render`: 0 errors; 6 warnings (R6-R8, R12, R13 are built only in new files; page 6,070 px) |
| Grounded in the code | class diagram members for `Job`, `R2ObjectStore`, `SubmitInputs`, `JobService`, `JobRepo`, `DetachedModalRemote`, `PipelineService`, `InputStore` checked against `main` at `fa131d4` by the lint |

Next: owner approval; the website spec (Next.js on Vercel) follows.
