"""spec-034 / spec-033 P-B: enforce bidirectional drift guard between dataclass and spec.

Rules:
  1. Every dataclass field on PipelineContext / _RunContext has exactly one
     row in specs/034-pipeline-context-contract/spec.md.
  2. Every spec row (in a "| `field_name` |" table cell) references a real
     dataclass field.

Fields explicitly exempted (instance methods, computed properties, etc.) are
listed in CONTEXT_EXEMPTIONS / RUN_CONTEXT_EXEMPTIONS.
"""
from __future__ import annotations

import re
from dataclasses import fields as dc_fields
from pathlib import Path

from face_cluster.pipeline import _RunContext
from sim_bench.pipeline.context import PipelineContext


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "specs" / "034-pipeline-context-contract" / "spec.md"


# Fields that exist on the dataclass but are deliberately not in the spec.
# Reason must be a one-line rationale.
CONTEXT_EXEMPTIONS: dict[str, str] = {
    # (none — every PipelineContext field is in spec-034)
}

RUN_CONTEXT_EXEMPTIONS: dict[str, str] = {
    # (none — every _RunContext field is in spec-034)
}


_RE_FIELD_CELL = re.compile(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`\s*\|", re.MULTILINE)
_RE_SECTION_3 = re.compile(r"^## 3\. ", re.MULTILINE)
_RE_SECTION_4 = re.compile(r"^## 4\. ", re.MULTILINE)
_RE_SECTION_5 = re.compile(r"^## 5\. ", re.MULTILINE)


def _spec_fields_by_section() -> tuple[set[str], set[str]]:
    """Return (PipelineContext fields, _RunContext fields) parsed from spec markdown.

    Section 3 covers PipelineContext; section 4 covers _RunContext; the
    boundary between them is the literal "## 4. " header.
    """
    src = SPEC.read_text(encoding="utf-8")
    s3 = _RE_SECTION_3.search(src)
    s4 = _RE_SECTION_4.search(src)
    s5 = _RE_SECTION_5.search(src)
    assert s3 and s4 and s5, "spec-034 must have sections 3, 4, 5"

    ctx_md = src[s3.end():s4.start()]
    run_md = src[s4.end():s5.start()]

    ctx_fields = set(_RE_FIELD_CELL.findall(ctx_md))
    run_fields = set(_RE_FIELD_CELL.findall(run_md))
    return ctx_fields, run_fields


def _dataclass_field_names(cls) -> set[str]:
    return {f.name for f in dc_fields(cls)}


def test_spec_exists():
    assert SPEC.exists(), f"spec-034 missing at {SPEC}"


def test_pipeline_context_fields_covered():
    """Every PipelineContext field has a spec row (or is exempted)."""
    spec_ctx, _ = _spec_fields_by_section()
    code_ctx = _dataclass_field_names(PipelineContext)

    missing = sorted((code_ctx - spec_ctx) - set(CONTEXT_EXEMPTIONS))
    assert not missing, (
        "PipelineContext fields without a spec-034 row:\n  "
        + "\n  ".join(missing)
        + "\nAdd a row to specs/034-pipeline-context-contract/spec.md or "
        "an entry to CONTEXT_EXEMPTIONS with a one-line reason."
    )


def test_run_context_fields_covered():
    """Every _RunContext field has a spec row (or is exempted)."""
    _, spec_run = _spec_fields_by_section()
    code_run = _dataclass_field_names(_RunContext)

    missing = sorted((code_run - spec_run) - set(RUN_CONTEXT_EXEMPTIONS))
    assert not missing, (
        "_RunContext fields without a spec-034 row:\n  "
        + "\n  ".join(missing)
        + "\nAdd a row to specs/034-pipeline-context-contract/spec.md or "
        "an entry to RUN_CONTEXT_EXEMPTIONS with a one-line reason."
    )


def test_spec_rows_reference_real_fields():
    """Every spec row's field name must exist on the dataclass it lives under."""
    spec_ctx, spec_run = _spec_fields_by_section()
    code_ctx = _dataclass_field_names(PipelineContext)
    code_run = _dataclass_field_names(_RunContext)

    stale_ctx = sorted(spec_ctx - code_ctx)
    stale_run = sorted(spec_run - code_run)

    errors = []
    if stale_ctx:
        errors.append(
            "spec-034 section 3 rows reference fields not on PipelineContext:\n  "
            + "\n  ".join(stale_ctx)
        )
    if stale_run:
        errors.append(
            "spec-034 section 4 rows reference fields not on _RunContext:\n  "
            + "\n  ".join(stale_run)
        )
    assert not errors, "\n\n".join(errors)


def test_exemptions_have_reasons():
    """Every exempted field needs a non-empty reason."""
    bad = []
    for k, v in CONTEXT_EXEMPTIONS.items():
        if not v or not v.strip():
            bad.append(f"CONTEXT_EXEMPTIONS[{k!r}] has empty reason")
    for k, v in RUN_CONTEXT_EXEMPTIONS.items():
        if not v or not v.strip():
            bad.append(f"RUN_CONTEXT_EXEMPTIONS[{k!r}] has empty reason")
    assert not bad, "\n".join(bad)
