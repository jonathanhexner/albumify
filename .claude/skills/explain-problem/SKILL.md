---
name: explain-problem
description: Use ONLY when the user explicitly asks you to explain a bug, a technical problem, or how something works across modules/layers (e.g. "explain this bug", "help me understand why X is broken", "walk me through how the data flows", "trace where this breaks"). Produces a paced, jargon-free interactive explanation, then a self-contained HTML report, then a short narration script. Do NOT auto-trigger after diagnosing a bug; wait to be asked.
version: 1.0.0
user-invocable: true
argument-hint: "[the bug / system / question to explain]"
---

# Explain a technical problem — clearly

The reader is an engineer who does **not** know this subsystem. **If they can't follow
it, the explanation failed — no matter how correct it is.** Clarity is the only goal that
matters here. (This skill exists because a "correct" explanation that was long, abstract,
and jargon-heavy left the reader understanding nothing.)

## The deliverable, in order

1. **Interactive explanation in chat** — paced, one layer at a time (default).
2. **A self-contained HTML report** — only after the explanation has landed.
3. **A ~60–90s narration script** (plain text) — the problem spoken aloud. No auto audio;
   tell the user they can run it through TTS if they want (e.g. Windows `System.Speech`).

## HOW to explain (process rules — these are the point)

- **Simple first, depth on ask.** Lead with the simplest correct framing. Add detail only
  when the user asks for it. Don't pre-empt.
- **One layer at a time. Check it landed before moving on.** Explain one component/stage,
  then stop and confirm ("clear? want the next one?"). Let the user drive the pace and the
  order. Do not dump the whole chain at once.
- **No undefined jargon.** Never use a technical term without a one-clause inline
  definition. Banned unless defined on the spot: "DTO", "boundary contract", "anemic",
  "projection", "coercion", etc. Prefer a plain word.
- **Don't repeat yourself.** If something was already explained or defined earlier in the
  conversation, do not re-explain or re-define it. Reference it.
- **Concise + precise.** Short chat answers. When the user says "short", be short. Defer
  anything long to the HTML, not the chat.
- **Evidence over speculation.** Ground every claim in real inspection (read/grep/run),
  not guesswork. **Verify a root cause before asserting it as fact** — confirm it with a
  concrete check (e.g. run the code path, print the dropped field). If you haven't verified
  yet, say so and label it a hypothesis with your confidence.
- **Say when you don't know.** State unknowns plainly instead of bluffing or filling gaps.

## WHAT every explanation must contain (chat + HTML)

- **Short background** for someone unfamiliar — the 2–3 sentence "what is this system" so
  they're not lost.
- **Symptom** — what is actually observed (the broken behaviour / the question).
- **Cause in one line** — the single-sentence root cause.
- **A plain analogy** for the core mechanism (the one that worked here: *"a form with only
  10 blanks — anything not on the form gets thrown away"*).
- **Each component named with its `file/path.py:line`** so the reader can open it
  themselves — plus that component's **purpose (why it exists)**, not just what it does.
- **The trace** — at each stage, does the data/behaviour **survive or break?** Mark the
  single **break point**.
- **Smoking-gun evidence** — the concrete proof of the cause (e.g. "the columns that work
  are exactly the fields the schema declares").
- **Bad vs good** — what's wrong now vs what good would look like (minimal code).
- **2–3 line senior-engineer take** — the design-level read.
- **Files in one place** — a closing table: object | module path | one-line purpose.

## The HTML report (locked structure)

Self-contained: inline CSS, no external dependencies. Skimmable in ~60 seconds. Shorter is
better. Sections, in order:

1. Symptom (callout)
2. Cause in one line (accent callout)
3. Plain analogy
4. Short background (who's this for / what is the system)
5. Numbered stage boxes top→bottom with arrows — **green border = survives, red = breaks**,
   a distinct **"THE BREAK POINT"** marker on the failing stage. Each box: number, name,
   `file:line`, one-line purpose, survive/break verdict.
6. Smoking-gun proof
7. A value-trace table: pick 2–3 example values, show survive/drop across the stages
   (green/red pills)
8. Bad-vs-good (two columns, minimal code)
9. 2–3 line senior-engineer take
10. Files-in-one-place table

Write it under `docs/architecture/` (or where the user points). Audience: engineers.

## Narration script

A ~60–90 second spoken-word version: symptom → cause → analogy, in plain sentences, no code
read-out. Save as `*_narration.md` next to the HTML. One line at the end: "Run through any
TTS for audio (e.g. PowerShell `System.Speech`)."

## Length budget

Chat: short, paced turns. HTML / any doc: ≤ ~200 lines. Practice what this skill preaches —
if your own explanation is long and abstract, it's wrong.

## Anti-patterns (these are exactly what failed before)

- Dumping a long, abstract wall up front instead of pacing it.
- Jargon with no definition.
- **Asserting a root cause before verifying it** (naming the wrong module confidently).
- Jumping ahead instead of going layer by layer at the user's pace.
- Re-explaining/re-defining something already covered.
