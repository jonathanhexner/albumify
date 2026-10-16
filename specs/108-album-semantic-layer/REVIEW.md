# spec-108 — Code Review (Album Semantic Layer)

**Reviewed:** 2026-10-02 · **Reviewer:** Claude · **Base:** `main`
**Scope:** P0 (semantic tags) + P1 (tag-based album theme) + P2 (per-keeper & album captions + the
isolated `.venv-vlm` VLM seam). P1.3 (VLM-refined theme), P3 (theme-relevance), P4 (scene-cluster
boost) are NOT built — out of scope.
**Verdict:** **spec stays In Progress.** P0 + P1 **ACCEPT** (done, verified). P2 **ACCEPT as
code-complete, NOT Implemented** — no live VLM E2E (deferred to the spec-109 GPU worker). No §1–§7
hard blockers, because the spec is *not* being flipped to Implemented.

---

## Part 1 — How it works

### Module inventory (all ≤116 LOC, single responsibility)
| File | LOC | Responsibility |
|---|---|---|
| `sim_bench/semantic/tags.py` | 82 | pure `clean()` + `categorize()` → `SemanticTags` (facets) |
| `sim_bench/semantic/theme.py` | 67 | pure `infer_theme()` → `AlbumTheme` (album_type, facets, motifs) |
| `sim_bench/semantic/captions.py` | 58 | pure `clean_caption()`, `build_album_prompt()`, `KEEPER_PROMPT` |
| `sim_bench/pipeline/steps/derive_semantic_tags.py` | 42 | step: `ram_tags` → `semantic_tags` |
| `sim_bench/pipeline/steps/infer_album_theme.py` | 41 | step: `semantic_tags` → `album_theme` |
| `sim_bench/pipeline/steps/caption_keepers.py` | 70 | step (opt-in): per-keeper VLM caption → `image_captions` |
| `sim_bench/pipeline/steps/caption_album.py` | 67 | step (opt-in): text-only synthesis → `album_theme.caption` |
| `sim_bench/vlm/base.py` | 38 | `VLMProvider` protocol (the seam) |
| `sim_bench/vlm/vlm_worker.py` | 116 | MiniCPM-V worker (runs in `.venv-vlm`), JSON-lines, `--stub` |
| `sim_bench/vlm/subprocess_provider.py` | 104 | `SubprocessVLMProvider` — spawns worker, implements seam |

### Dependency map / layering
`steps/*` (framework) → `sim_bench/semantic/*` + `sim_bench/vlm/*` (framework-agnostic domain). The VLM
backend is reachable only through the `VLMProvider` protocol — steps never import a concrete model. No
reverse imports; the main env never imports transformers-4.51 (that lives in `.venv-vlm`).

### Data flow
`triage_content` (ram_tags, cached) → `derive_semantic_tags` (semantic_tags) → `infer_album_theme`
(album_theme) → … → `select_best` → `caption_keepers` (image_captions) → `caption_album`
(album_theme.caption). Captions are opt-in (`enabled=false`) and run only on keepers.

---

## Part 2 — Findings by checklist section

### §1 Structure — **pass**
Every module small + single-purpose; steps thin (spec-053). No dead code. No >4-param functions.

### §2 Code quality — **pass**
`try/except` only around model load + per-image inference, each degrading to `""`/empty (never raises,
so one bad image can't fail a run). No silent defaults on load-bearing fields. Nesting ≤3.

### §3 Naming / package structure — **pass**
New `sim_bench/semantic/` + `sim_bench/vlm/` subpackages (≥3 related files) — correct promotion.
`__all__` present on the domain modules; `__init__.py` empty by repo convention; `vlm_worker.py` is a
script (no `__all__` needed).

### §4 Layering / coupling — **pass-with-followup**
Clean domain/step split; VLM behind a protocol. **Finding (low):** `album_theme` has **two writers** —
`infer_album_theme` produces it (`source="tags"`), `caption_album` overwrites it
(`source="vlm"` + caption). Intentional (refine-in-place) and ordered by `depends_on`, but a two-way
writer per §4 must be documented → ticket T1.

### §5 Testability — **P0/P1 pass · P2 pass-with-followup**
Inventory: **26 unit tests** (tags 8, theme 5, theme-step 2, derive-step 2, captions 12). P0/P1 are
pure deterministic logic over tags (no images needed) → unit tests are the right coverage; also
sanity-checked on real Slovakia tags. **Finding (P2, medium):** the caption steps are tested only with
a **fake provider** — there is **no live VLM E2E on real data** (MiniCPM-V loads but CPU inference is
impractical here; deferred to the GPU worker). The steps are gated OFF (`enabled=false`) and not in any
active production path, so this is a followup, not a blocker → ticket T3. The stub-verified subprocess
IPC proves the plumbing independent of the model.

### §6 Boundary contracts — **pass-with-followup**
No new Pydantic/Pandera boundary — internal dataclasses (`SemanticTags`, `AlbumTheme`) + context dicts.
`VLMProvider` is a `runtime_checkable` Protocol (the seam's contract) and `isinstance` holds for the
provider. New context fields (`semantic_tags`, `album_theme`) have spec-034 rows. **Finding (low):**
the `album_theme` dict shape is not type-enforced; promote to a typed model (`extra="forbid"`) when the
UI consumes it → ticket T4.

### §7 Documentation — **pass-with-followup**
`spec.md`, `tasks.md`, `REVIEW.md` (this), CHANGES_LOG (5 entries), LEARNINGS (VLM-on-CPU entry) all
present. **Finding (medium):** `docs/architecture/data_flow.html` + `classes.html` not yet updated for
the semantic-layer steps, the new context fields, and the VLM classes → ticket T2 (documentation
mandate).

### §8 Risk register — **pass**
Deferred work (P2 live, P1.3, P3, P4) tracked in tasks.md. VLM env friction + mitigations captured in
LEARNINGS. Backwards-compat: caption steps gated off → other pipelines/CI unaffected. Hot-path: P0/P1
reuse cached tags (near-zero cost); the VLM is the only heavy piece and it is gated + isolated.

---

## Part 3 — Verdict & tickets

### Verdict
- **P0 (semantic tags) — ACCEPT.** Done, verified, 10 tests.
- **P1 (tag-based theme) — ACCEPT.** Done, verified ("family trip — towns & nature"), 7 tests.
- **P2 (captions + VLM seam) — ACCEPT as code-complete; NOT Implemented** until a live caption E2E runs
  on the spec-109 GPU worker (T3).
- **spec-108 overall remains `In Progress`** (P1.3/P3/P4 unbuilt). No handoff to Implemented attempted,
  so no §1–§7 blocker gate applies.

### Follow-up tickets (→ TODO / spec-109)
- **T1 (§4, low):** document the `album_theme` two-writer contract (produce→refine) in code + spec-034.
- **T2 (§7, med):** update `data_flow.html` + `classes.html` for semantic steps, context fields, VLM classes.
- **T3 (§5, med):** live caption E2E (MiniCPM-V) on the spec-109 GPU worker — gates P2 → Implemented.
- **T4 (§6, low):** typed `extra="forbid"` model for `album_theme` (and `set_aside`) when the UI consumes them.
