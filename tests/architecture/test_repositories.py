"""spec-043 — architecture tests for the Repository layer.

Two invariants enforced at PR time:

1. Every ``*Repository`` class takes a typed Config dataclass in its
   ``__init__`` (or no args). Forbids growing kwarg lists.

2. ``face_cluster/repositories/*.py`` exposes only classes, dataclasses,
   and module-level constants — no module-level functions that query
   the DB. Forbids the legacy free-function shape.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_DIR = REPO_ROOT / "face_cluster" / "repositories"


def _repository_classes() -> list[type]:
    """Discover every class named ``*Repository`` in face_cluster.repositories.*."""
    import face_cluster.repositories as pkg
    out: list[type] = []
    # The package __init__ re-exports every Repository class.
    for name in pkg.__all__:
        obj = getattr(pkg, name, None)
        if isinstance(obj, type) and name.endswith("Repository"):
            out.append(obj)
    return out


def test_repository_classes_exist():
    """At least one *Repository class is exported (sanity)."""
    classes = _repository_classes()
    assert classes, (
        "No *Repository classes exported from face_cluster.repositories. "
        "The package is supposed to be the new persistence layer."
    )


def test_repositories_take_typed_config():
    """Each *Repository class's __init__ takes at most one positional/keyword
    argument (besides self) and that argument is a frozen dataclass
    (or Optional[frozen dataclass]).

    Forbids the 'add a 5th kwarg' anti-pattern that motivated the spec-043
    refactor in the first place.
    """
    import typing
    offenders: list[str] = []
    for cls in _repository_classes():
        sig = inspect.signature(cls.__init__)
        # Drop self.
        params = [
            p for p in sig.parameters.values()
            if p.name != "self"
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        if len(params) > 1:
            offenders.append(
                f"{cls.__name__}.__init__ takes {len(params)} args; "
                f"max is 1 (a Config dataclass). Found: {[p.name for p in params]}"
            )
            continue
        if not params:
            continue  # zero-arg constructor is fine (uses defaults)
        # Resolve string annotations via get_type_hints (handles
        # ``from __future__ import annotations`` lazy evaluation).
        try:
            hints = typing.get_type_hints(cls.__init__)
        except Exception as e:  # pragma: no cover
            offenders.append(f"{cls.__name__}: get_type_hints failed: {e}")
            continue
        ann = hints.get(params[0].name)
        if ann is None:
            offenders.append(
                f"{cls.__name__}.__init__'s {params[0].name!r} arg has no type annotation."
            )
            continue
        # Unwrap Optional[X] / X | None.
        if hasattr(ann, "__args__"):
            non_none = [a for a in ann.__args__ if a is not type(None)]
            if len(non_none) == 1:
                ann = non_none[0]
        if not (dataclasses.is_dataclass(ann)
                and getattr(ann, "__dataclass_params__", None) is not None
                and ann.__dataclass_params__.frozen):
            offenders.append(
                f"{cls.__name__}.__init__'s single arg is annotated "
                f"{ann!r}; must be a frozen dataclass."
            )
    assert not offenders, (
        "Repository __init__ violations:\n  - " + "\n  - ".join(offenders)
    )


def _module_level_function_names(path: Path) -> list[str]:
    """Return the names of top-level def statements in a module (public only)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    return [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def test_repositories_module_has_no_public_free_functions():
    """face_cluster/repositories/*.py exposes only classes, dataclasses,
    and constants. No public module-level functions.

    Private helpers (``_foo``) are allowed — they're not part of the
    Repository's contract.
    """
    if not REPO_DIR.exists():
        pytest.skip(f"{REPO_DIR} not present (spec-043 not landed yet?)")
    offenders: list[str] = []
    for path in sorted(REPO_DIR.glob("*.py")):
        # Skip __init__.py and underscore-prefixed infrastructure modules
        # (_engine.py, _session.py, _orm_base.py, _base_repository.py, _errors.py).
        # Those are private infra primitives reused by the Repository classes;
        # the rule targets DB-query free functions in public modules.
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        funcs = _module_level_function_names(path)
        if funcs:
            offenders.append(f"{path.name}: {funcs}")
    assert not offenders, (
        "Module-level public functions found in face_cluster/repositories/. "
        "Persistence access must go through Repository classes, not free "
        "functions:\n  - " + "\n  - ".join(offenders)
    )
