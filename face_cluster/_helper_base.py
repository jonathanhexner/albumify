"""spec-053 — Pipeline helper protocol.

A "helper" is a domain class in ``face_cluster/`` that performs one
bounded calculation: quality gating, k-NN graph building, exemplar
selection, merging, exporting. Pipeline steps in
``sim_bench/pipeline/steps/`` are thin wrappers over these helpers.

Convention (locked in spec-053):

- **Config goes in ``__init__``.** Helper instances are bound to a
  configuration. This matches the existing pattern across all five
  helpers (``QualityGater``, ``KNNGraphBuilder``, ``ExemplarSelector``,
  ``SimplifiedMerger``, ``RunExporter``).
- **Per-call data goes in ``Inputs``** — a small typed dataclass with
  ONLY what the helper needs. NOT a ``PipelineContext`` (helpers must
  remain framework-agnostic so notebooks / analyses can use them
  directly).
- **Output is a typed ``Result`` dataclass** — only what callers need.
- **Pipeline steps call exactly one method: ``calc(inputs)``.** They
  translate ``PipelineContext`` ↔ ``Inputs`` / ``Result``. Individual
  helper methods (``compute_blur_scores``, ``build_graph`` …) stay
  public for non-pipeline callers but pipeline steps MUST use
  ``calc()`` so the orchestration constraints (e.g. "compute blur
  before selecting core set") are enforced by construction.

Why this exists: prior to spec-053 two quality-gate steps wrapped
``QualityGater`` differently; one forgot to call ``compute_blur_scores``
before ``select_core_set`` and the blur threshold silently self-disabled
in production. A single ``calc()`` entry point makes that class of
bug impossible by construction.
"""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

I = TypeVar("I")  # Inputs dataclass type — per-call data
R = TypeVar("R")  # Result dataclass type — helper output


@runtime_checkable
class PipelineHelper(Protocol[I, R]):
    """Structural type for a pipeline helper.

    Helpers don't need to inherit from this — Python's structural typing
    matches anything with a compatible ``calc`` method. ``@runtime_checkable``
    lets tests assert ``isinstance(helper, PipelineHelper)``.
    """

    def calc(self, inputs: I) -> R: ...


__all__ = ["PipelineHelper"]
