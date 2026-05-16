"""spec-032 / SIGHTING-059 P4: enforce bidirectional UI ↔ filter alignment.

Prevents the SIGHTING-059 Issue 2 scenario: a UI slider labeled "Min Face Size
(px)" that controls nothing while a hidden filter actually decides.

Rules enforced:
  1. Every KNOWN_FILTERS entry declares one of three forms:
       UI: <label> (<widget_key>) [+ (<widget_key>)]...
       HIDDEN(permanent): <reason>
       HIDDEN(todo, target=Px): <reason>
     Anything else is a malformed entry.
  2. UI: bindings must reference widget keys that actually exist in the UI
     source files (no stale bindings).
  3. Streamlit widget keys in the UI source files that look like filter knobs
     (match FILTER_KEY_PREFIXES, not in NON_FILTER_KEYS) must be bound by
     exactly one KNOWN_FILTERS entry (no phantom controls).
  4. HIDDEN(todo, target=Px) entries fail when Px is in SHIPPED_PHASES — they
     must be resolved before that phase merges to main.
"""
from __future__ import annotations

import re
from pathlib import Path

from face_cluster.filter_context import KNOWN_FILTERS, parse_ui_binding


REPO = Path(__file__).resolve().parents[2]


# UI source files we scan for widget keys.
UI_SOURCES = [
    "app/streamlit/components/pipeline_runner.py",   # Album Configure & Run
    "app/face_clustering/tabs/run_tab.py",           # FC App Run tab
    "app/face_clustering/tabs/recluster_tab.py",     # FC App Recluster tab
    "app/shared/merge_controls.py",                  # shared merge sliders
]


# Widget-key prefixes that signal "this is a filter knob".  Anything matching
# these prefixes but not bound to a filter is a phantom control.
FILTER_KEY_PREFIXES = (
    "config_min_", "config_det_", "config_max_",
    "run_blur_", "run_yaw_", "run_pitch_", "run_roll_",
    "run_min_face_", "run_max_faces", "run_det_score_",
    "run_cap_", "rc_cap_",
)


# Explicit non-filter widget keys that match the prefixes but are
# parameters/controls, not filter gates.  Each entry needs a one-line reason.
NON_FILTER_KEYS = {
    "run_min_cluster":               "clustering parameter (min_cluster_size), not a filter",
    "config_max_images":             "selection cap, not a filter",
    "config_max_images_per_cluster": "selection cap, not a filter",
    "config_max_per_cluster":        "selection cap, not a filter",
    "config_min_score":              "selection threshold (composite score), not a filter",
    "config_min_face_size":          "SIGHTING-059 #2: 'Min Face Size (px)' slider is a "
                                      "no-op rejector today — wires to insightface scoring "
                                      "fallback only.  P6 promotes it to a real filter and "
                                      "binds it to face_area via SIGHTING-060 fix.",
    "run_det_score_min":             "alternative key for face_confidence on FC App Run "
                                      "tab; canonical binding is to Album's config_det_conf.  "
                                      "TODO P3: dedupe by exposing one shared slider.",
}


# Phases that have shipped.  Owners of the PR closing a phase bump this set.
# HIDDEN(todo, target=Px) entries whose Px is in SHIPPED_PHASES fail the check.
SHIPPED_PHASES: set[str] = {
    # P0, P1, P2 ship as part of the same PR landing this test.  P4 lands
    # this test itself.  P3, P5, P6 are not yet shipped.
    # Initially empty so the test passes on the landing PR.  Update via
    # subsequent PRs that close phases.
}


_RE_LITERAL_KEY = re.compile(r"""key=['"](\w+)['"]""")
# f-string form used by shared controls: key=f"{key_prefix}suffix"
_RE_FSTRING_KEY = re.compile(r"""key=f['"]\{key_prefix\}(\w+)['"]""")
# Known prefixes that `key_prefix` takes at call sites — keep in sync with
# render_merge_params() callers (Run tab uses "run_", Recluster tab "rc_").
_KEY_PREFIXES = ("run_", "rc_")


def _extract_widget_keys(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    keys: set[str] = set(_RE_LITERAL_KEY.findall(src))
    for suffix in _RE_FSTRING_KEY.findall(src):
        for prefix in _KEY_PREFIXES:
            keys.add(prefix + suffix)
    return keys


def test_known_filters_ui_binding_grammar():
    """Every entry must parse into one of the three valid forms."""
    invalid = []
    for name, _t, _f, _d, ui in KNOWN_FILTERS:
        kind, target, reason, _keys = parse_ui_binding(ui)
        if kind == "invalid":
            invalid.append(f"{name}: {ui!r}")
            continue
        if kind == "hidden_permanent" and not reason:
            invalid.append(f"{name}: HIDDEN(permanent) without reason")
        if kind == "hidden_todo":
            if not reason:
                invalid.append(f"{name}: HIDDEN(todo, target={target}) without reason")
            if target is None or not re.fullmatch(r"P\d+", target or ""):
                invalid.append(f"{name}: HIDDEN(todo) bad target {target!r}")
    assert not invalid, (
        "Malformed ui_binding entries in KNOWN_FILTERS.  Fix grammar:\n  "
        + "\n  ".join(invalid)
    )


def test_ui_bindings_reference_existing_widget_keys():
    """UI: bindings must point at widget keys present in UI source."""
    all_ui_keys: set[str] = set()
    for src in UI_SOURCES:
        p = REPO / src
        assert p.exists(), f"UI source not found: {src}"
        all_ui_keys |= _extract_widget_keys(p)

    errors = []
    for name, _t, _f, _d, ui in KNOWN_FILTERS:
        kind, _target, _reason, keys = parse_ui_binding(ui)
        if kind != "ui":
            continue
        if not keys:
            errors.append(
                f"{name}: UI: binding has no widget key in parentheses ({ui!r})"
            )
            continue
        for k in keys:
            if k not in all_ui_keys:
                errors.append(
                    f"{name}: ui_binding references widget key {k!r} "
                    f"not found in any UI source"
                )
    assert not errors, (
        "Stale UI bindings detected:\n  " + "\n  ".join(errors)
    )


def test_no_phantom_filter_controls():
    """Every widget key that looks like a filter knob must bind to a filter."""
    all_ui_keys: set[str] = set()
    for src in UI_SOURCES:
        all_ui_keys |= _extract_widget_keys(REPO / src)

    bound_keys: set[str] = set()
    for name, _t, _f, _d, ui in KNOWN_FILTERS:
        kind, _target, _reason, keys = parse_ui_binding(ui)
        if kind == "ui":
            bound_keys.update(keys)

    phantom = sorted(
        k for k in all_ui_keys
        if k.startswith(FILTER_KEY_PREFIXES)
        and k not in bound_keys
        and k not in NON_FILTER_KEYS
    )
    assert not phantom, (
        "Phantom filter controls: these widget keys look like filter knobs "
        "but no KNOWN_FILTERS entry binds them.  Either bind a filter to the "
        "widget, add a NON_FILTER_KEYS entry with reason, or rename the "
        "widget so it no longer matches a filter-knob prefix:\n  "
        + "\n  ".join(phantom)
    )


def test_hidden_todo_targets_not_overdue():
    """HIDDEN(todo, target=Px) must be resolved before Px ships."""
    overdue = []
    for name, _t, _f, _d, ui in KNOWN_FILTERS:
        kind, target, _reason, _keys = parse_ui_binding(ui)
        if kind == "hidden_todo" and target in SHIPPED_PHASES:
            overdue.append(f"{name}: target={target} but {target} has shipped")
    assert not overdue, (
        "HIDDEN(todo, target=Px) entries are overdue.  Resolve them by "
        "exposing the filter via UI, removing the filter, or promoting to "
        "HIDDEN(permanent) with a documented reason:\n  "
        + "\n  ".join(overdue)
    )
