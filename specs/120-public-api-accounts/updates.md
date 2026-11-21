# spec-120 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress 2026-11-22 (owner approved; W1 done)

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
