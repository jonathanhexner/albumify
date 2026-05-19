"""spec-040 B7: chain order has one source of truth.

The unified clustering chain is declared in two places:

* ``UNIFIED_CLUSTERING_STEPS`` literal in ``face_cluster.fc_app_runner`` —
  the explicit ordered list FCAppRunner passes to PipelineExecutor.
* ``depends_on`` metadata on each ``BaseStep`` in
  ``sim_bench.pipeline.steps.face_clustering_steps`` — used by
  ``PipelineExecutor`` when called with ``auto_resolve=True``.

Both are load-bearing: the literal is the readable canonical reference;
the metadata is required for ad-hoc step invocation by name. REVIEW.md B7
called this out as a drift risk. This test resolves it by enforcing that
the literal IS a valid topological ordering of the depends_on graph — if
either side is edited without the other, the test breaks.
"""
from __future__ import annotations

from typing import Dict, List, Set


def _topological_sort(deps: Dict[str, List[str]]) -> List[str]:
    """Kahn's algorithm; raises on cycles. Deterministic via lexicographic tiebreak."""
    in_degree: Dict[str, int] = {n: 0 for n in deps}
    for node, prereqs in deps.items():
        for p in prereqs:
            if p not in in_degree:
                in_degree[p] = 0
            in_degree[node] = in_degree.get(node, 0) + 1
    ready: List[str] = sorted(n for n, d in in_degree.items() if d == 0)
    result: List[str] = []
    rev: Dict[str, List[str]] = {n: [] for n in in_degree}
    for node, prereqs in deps.items():
        for p in prereqs:
            rev[p].append(node)
    while ready:
        n = ready.pop(0)
        result.append(n)
        for downstream in sorted(rev.get(n, [])):
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                ready.append(downstream)
        ready.sort()
    if len(result) != len(in_degree):
        raise ValueError(f"cycle in depends_on graph: {deps}")
    return result


def test_unified_clustering_steps_literal_matches_depends_on_graph():
    """The literal in fc_app_runner must match a valid topo sort of depends_on.

    If a future edit reorders one without the other, this test fails with
    a side-by-side diff so the drift is caught at CI rather than at first
    real-album run.
    """
    import sim_bench.pipeline.steps.all_steps  # noqa: F401  -- register steps

    from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
    from sim_bench.pipeline.registry import get_registry

    registry = get_registry()
    deps: Dict[str, List[str]] = {}
    for name in UNIFIED_CLUSTERING_STEPS:
        step = registry.get(name)
        # Restrict the dep graph to the unified chain — depends_on may
        # reference upstream producer steps (detect_persons etc.) that
        # are not part of the clustering chain itself.
        in_chain = [d for d in step.metadata.depends_on if d in set(UNIFIED_CLUSTERING_STEPS)]
        deps[name] = in_chain

    topo = _topological_sort(deps)
    assert UNIFIED_CLUSTERING_STEPS == topo, (
        "UNIFIED_CLUSTERING_STEPS literal in face_cluster.fc_app_runner has drifted "
        "from the depends_on graph declared on the unified clustering steps.\n"
        f"  Literal: {UNIFIED_CLUSTERING_STEPS}\n"
        f"  Topo:    {topo}\n"
        "Either update the literal to match the graph, or update each step's "
        "depends_on to reflect the new intended order."
    )


def test_unified_clustering_steps_form_a_linear_chain():
    """Each step (except the first) depends on exactly one prior step in the chain.

    Reading the code assumes a single linear chain — any future branching would
    invalidate FCAppRunner's "run the list in order" approach. Flag it here.
    """
    import sim_bench.pipeline.steps.all_steps  # noqa: F401

    from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
    from sim_bench.pipeline.registry import get_registry

    registry = get_registry()
    chain: Set[str] = set(UNIFIED_CLUSTERING_STEPS)
    for i, name in enumerate(UNIFIED_CLUSTERING_STEPS):
        step = registry.get(name)
        in_chain_deps = [d for d in step.metadata.depends_on if d in chain]
        if i == 0:
            assert in_chain_deps == [], (
                f"first step {name!r} has in-chain depends_on={in_chain_deps}; "
                "expected none."
            )
        else:
            expected = UNIFIED_CLUSTERING_STEPS[i - 1]
            assert in_chain_deps == [expected], (
                f"step {name!r} should depend on {expected!r} (the immediate "
                f"predecessor in UNIFIED_CLUSTERING_STEPS) but its in-chain "
                f"depends_on is {in_chain_deps}."
            )
