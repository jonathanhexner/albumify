"""scripts/prd_lint.py: PRD format limits, named checks, plan coverage, file tables, and class-diagram members vs code.

The class-diagram case is the one the 2026-10-23 review found by hand: methods drawn as "changed" (✎) that did
not exist in the code yet.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("prd_lint", Path(__file__).parent.parent / "scripts" / "prd_lint.py")
prd_lint = importlib.util.module_from_spec(_spec)
sys.modules["prd_lint"] = prd_lint   # dataclasses look their module up here
_spec.loader.exec_module(prd_lint)

CODE = '''
class Store:
    kind = "x"

    def __init__(self):
        self.root = None

    def put(self, key):
        return key
'''


def _spec_md(objective="Do one thing.", reqs=None, plan="R1", diagram=None, new_rows=None, mod_rows=None):
    reqs = reqs if reqs is not None else ["| R1 | ○ | it works | `tests/test_store.py::test_put` |"]
    diagram = diagram if diagram is not None else "class Store {\n    +put(key) str\n    ✚ get(key) str\n  }"
    new_rows = new_rows if new_rows is not None else ["| `sim_bench/new_mod.py` | `New` | does | `a` | `b` | ○ W1 |"]
    mod_rows = mod_rows if mod_rows is not None else ["| `sim_bench/store.py` | `Store` | old | new | R1 | ○ W1 |"]
    return "\n".join([
        "# spec-999 — Test", "", '<p class="legend">legend</p>', "",
        "## 1. Objective", "", objective, "",
        "## 2. Requirements", "", "| # | Status | Requirement | How we know it works |", "|---|---|---|---|", *reqs, "",
        "## 3. Plan", "", "```mermaid", f'flowchart TD\n  A["step"] --> C{{{{"check {plan}"}}}}', "```", "",
        "## 4. Class diagram", "", "```mermaid", "classDiagram", f"  {diagram}", "```", "",
        "## 5. New files", "", "| File | Class | Responsibility | Methods | Attributes | Status |", "|---|---|---|---|---|---|",
        *new_rows, "",
        "## 6. Modified files", "", "| File | Class / function | Today | After this spec | Req | Status |",
        "|---|---|---|---|---|---|", *mod_rows, "",
    ])


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "sim_bench").mkdir()
    (tmp_path / "sim_bench" / "store.py").write_text(CODE, encoding="utf-8")
    return tmp_path


def _lint(repo, **kw):
    spec = repo / "spec.md"
    spec.write_text(_spec_md(**kw), encoding="utf-8")
    return prd_lint.lint(spec, root=repo)


class ut_PrdLint:

    def test_a_well_formed_spec_passes(self, repo):
        res = _lint(repo)
        assert res.errors == []

    def test_objective_and_requirement_limits(self, repo):
        reqs = [f"| R{i} | ○ | r | real run |" for i in range(1, 17)]
        res = _lint(repo, objective="a\nb\nc\nd", reqs=reqs, plan="R1-R16")
        assert any("objective is 4 lines" in e for e in res.errors)
        assert any("16 requirements" in e for e in res.errors)

    def test_a_bare_check_and_a_missing_status_fail(self, repo):
        res = _lint(repo, reqs=["| R1 | planned | it works | tests |"])
        assert any("status must start" in e for e in res.errors)
        assert any("must name a test" in e for e in res.errors)

    def test_every_requirement_must_close_in_the_plan(self, repo):
        reqs = ["| R1 | ○ | a | real run |", "| R2 | ○ | b | real run |"]
        res = _lint(repo, reqs=reqs, plan="R1")
        assert any("R2 is not closed" in e for e in res.errors)

    def test_plan_ranges_count(self, repo):
        reqs = [f"| R{i} | ○ | r | real run |" for i in (1, 2, 3)]
        assert not any("not closed" in e for e in _lint(repo, reqs=reqs, plan="R1-R3").errors)

    def test_changed_member_that_does_not_exist_is_an_error(self, repo):
        diagram = "class Store {\n    ✎ spawn(job) str\n    ✖ gone()\n    +root: Path\n    ✎ put(key)\n  }"
        errors = _lint(repo, diagram=diagram).errors
        assert any("Store.spawn is marked changed" in e for e in errors)
        assert any("Store.gone is marked removed" in e for e in errors)
        assert not any("root" in e or ".put" in e for e in errors)   # self.root and put exist

    def test_a_class_not_in_the_code_may_only_have_new_members(self, repo):
        errors = _lint(repo, diagram="class Fresh {\n    ✚ calc()\n    ✎ other()\n  }").errors
        assert errors == ["class diagram: Fresh is not in the code, but 'other' is marked changed (mark new members with ✚)"]

    def test_pages_and_routers_are_not_checked_against_python(self, repo):
        assert _lint(repo, diagram="class JobsPage {\n    <<page>>\n    run form\n  }").errors == []

    def test_file_tables_match_the_disk(self, repo):
        res = _lint(repo, new_rows=["| `sim_bench/store.py` | `Store` | d | m | a | ○ W1 |"],
                    mod_rows=["| `sim_bench/missing.py` | `X` | o | n | R1 | ○ |"])
        assert any("new file sim_bench/store.py already exists" in e for e in res.errors)
        assert any("modified file sim_bench/missing.py does not exist" in e for e in res.errors)

    def test_brace_paths_expand(self):
        assert prd_lint._paths("`docs/{a,b}.html`") == ["docs/a.html", "docs/b.html"]

    def test_spec_115_passes(self):
        """The first spec in this format stays lint-clean."""
        spec = Path(__file__).parent.parent / "specs" / "115-album-workbench" / "spec.md"
        assert prd_lint.lint(spec).errors == []
