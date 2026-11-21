# spec-120 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Draft 2026-11-22 (awaiting owner approval)

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
