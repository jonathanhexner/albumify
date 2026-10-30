"""Compare page (spec-114): two runs on the same photos, A = reference, B = new.

A regression album preselects its baseline as A and its newest check as B; "Any two jobs" compares
whatever you pick (CPU vs L4, before vs after a settings change). Every photo opens a dialog with what
A and B each did with it. Clickable photos are real st.button grids (never dataframe row-select).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

from app.streamlit.api_client import ApiError, get_client

logger = logging.getLogger(__name__)

COLS = 6
PAGE = 36
ANY = "Any two jobs"
STATE_LABEL = {"pick": "Kept", "near_dup": "Near-duplicate", "not_selected": "Not selected",
               "set_aside": "Set aside", "removed": "Removed by a filter"}


def render_compare_page() -> None:
    st.title("Compare runs")
    client = get_client()
    try:
        jobs = [j for j in client.list_jobs(limit=1000) if j["status"] == "succeeded"]
    except ApiError as e:
        st.error(f"Cannot list jobs: {e}")
        return
    pair = _choose_pair(jobs)
    if pair is None:
        return
    a, b = pair
    try:
        cmp = _compare(client, a, b)
    except ApiError as e:
        st.error(str(e))
        return
    _render_verdict(cmp)
    _render_settings_diff(client, a, b)
    _render_checks(cmp)
    _render_sections(client, cmp)
    photo_ids = st.session_state.get("compare_photo")
    if photo_ids:
        _photo_dialog(client, cmp, photo_ids)


def _dataset(job: Dict) -> Optional[str]:
    tag = job.get("tag") or ""
    return tag.split(":")[1] if tag.startswith("regression:") else None


def _is_baseline(job: Dict) -> bool:
    return (job.get("tag") or "").endswith(":baseline")


def _label(job: Dict) -> str:
    base = "  [baseline]" if _is_baseline(job) else ""
    return f"{job['id']}{base}  ({job['runner']}, {job['n_inputs']} photos)"


def _choose_pair(jobs: List[Dict]) -> Optional[Tuple[str, str]]:
    datasets = sorted({d for d in map(_dataset, jobs) if d})
    choice = st.selectbox("Regression album", datasets + [ANY], key="compare_dataset",
                          help="A regression album compares its runs with the saved baseline.")
    pool = jobs if choice == ANY else [j for j in jobs if _dataset(j) == choice]
    if len(pool) < 2:
        st.info("Need at least two finished jobs on the same photos to compare.")
        return None
    pool = sorted(pool, key=lambda j: (not _is_baseline(j), j["created_at"]))   # baseline first
    ids = [j["id"] for j in pool]
    labels = {j["id"]: _label(j) for j in pool}
    newest = max(pool, key=lambda j: j["created_at"])["id"]
    c1, c2 = st.columns(2)
    a = c1.selectbox("A - reference", ids, index=0, format_func=labels.get, key=f"compare_a_{choice}")
    rest = [i for i in ids if i != a]
    key_b = f"compare_b_{choice}"
    # a re-run preselects B through session state (spec-115 R5); a default index as well makes Streamlit warn
    default_b = {} if st.session_state.get(key_b) in rest else {
        "index": rest.index(newest) if newest in rest else len(rest) - 1}
    b = c2.selectbox("B - new run", rest, format_func=labels.get, key=key_b, **default_b)
    return a, b


def _compare(client, a: str, b: str) -> Dict:
    """One API call per pair; kept in the session so clicking photos does not refetch."""
    cache = st.session_state.setdefault("compare_cache", {})
    if (a, b) not in cache:
        cache[(a, b)] = client.compare_jobs(b, a)
    return cache[(a, b)]


def _render_settings_diff(client, a: str, b: str) -> None:
    """spec-115 R12: what the two runs ran with, where it differs."""
    from sim_bench.serving.run_settings import RunSettings

    sa = RunSettings.from_dict(client.get_job(a).get("settings"))
    sb = RunSettings.from_dict(client.get_job(b).get("settings"))
    with st.expander("Settings that differ", expanded=False):
        if sa is None or sb is None:
            st.caption("One of these runs predates recorded settings (spec-115).")
            return
        diff = sa.diff(sb)
        if not diff:
            st.caption("Same pipeline, parameters, runner, device, code and libraries.")
            return
        st.table([{"setting": k, "A - reference": str(v[0]), "B - new run": str(v[1])} for k, v in diff.items()])


def _render_verdict(cmp: Dict) -> None:
    m, p = cmp["metrics"], cmp["picks"]
    if cmp["ok"]:
        st.success("Close enough: every count and the picks are within the allowed difference.")
    else:
        st.error("Not close enough:\n\n" + "\n".join(f"- {f}" for f in cmp["failures"]))
    cols = st.columns(6)
    for col, (label, value) in zip(cols, [
            ("Same photo", len(p["same"])), ("Swapped", len(p["swapped"])),
            ("Missing", len(p["missing"])), ("Extra", len(p["extra"])),
            ("Pick recall", f"{m['pick_recall']:.2f}"), ("Pick precision", f"{m['pick_precision']:.2f}")]):
        col.metric(label, value)


def _render_checks(cmp: Dict) -> None:
    with st.expander("Counts and people sizes (A, B, allowed difference)"):
        st.table([{"Check": c["name"], "A": c["baseline"], "B": c["current"],
                   "Difference": f"{c['current'] - c['baseline']:+d}", "Allowed": f"+/-{c['allowed']}",
                   "": "ok" if c["ok"] else "OUT OF BAND"} for c in cmp["checks"]])


def _render_sections(client, cmp: Dict) -> None:
    p = cmp["picks"]
    tabs = st.tabs([f"Missing ({len(p['missing'])})", f"Extra ({len(p['extra'])})",
                    f"Swapped ({len(p['swapped'])})", f"More from a group ({len(p['also_kept'])})",
                    f"People ({len(cmp['people'][cmp['b']])})"])
    with tabs[0]:
        st.caption("A kept this moment; B kept nothing from its near-duplicate group.")
        _grid(client, cmp["a"], p["missing"], "missing")
    with tabs[1]:
        st.caption("B kept a moment A did not keep.")
        _grid(client, cmp["b"], p["extra"], "extra")
    with tabs[2]:
        st.caption("Same moment, different shot: A's pick (left) and B's pick (right).")
        _pairs(client, cmp, p["swapped"])
    with tabs[3]:
        st.caption("B kept a second shot of a moment A also kept.")
        _grid(client, cmp["b"], p["also_kept"], "also")
    with tabs[4]:
        _people(client, cmp)


def _paged(items: List, key: str) -> List:
    if len(items) <= PAGE:
        return items
    n = (len(items) + PAGE - 1) // PAGE
    page = st.number_input(f"Page (1-{n})", 1, n, 1, key=key)
    return items[(page - 1) * PAGE: page * PAGE]


def _open(photo_ids: List[str]) -> None:
    st.session_state["compare_photo"] = photo_ids
    st.rerun()


def _grid(client, job_id: str, photo_ids: List[str], key: str) -> None:
    if not photo_ids:
        st.success("None.")
        return
    shown = _paged(photo_ids, f"compare_page_{key}")
    for row in range(0, len(shown), COLS):
        for col, pid in zip(st.columns(COLS), shown[row:row + COLS]):
            with col:
                st.image(client.job_photo_url(job_id, pid), use_container_width=True)
                if st.button(pid, key=f"cmp_{key}_{pid}", use_container_width=True):
                    _open([pid])


def _pairs(client, cmp: Dict, pairs: List[List[str]]) -> None:
    if not pairs:
        st.success("None.")
        return
    shown = _paged(pairs, "compare_page_swapped")
    for row in range(0, len(shown), COLS // 2):
        cols = st.columns(COLS)
        for i, (a_pid, b_pid) in enumerate(shown[row:row + COLS // 2]):
            for col, job, pid, side in ((cols[2 * i], cmp["a"], a_pid, "A"),
                                        (cols[2 * i + 1], cmp["b"], b_pid, "B")):
                with col:
                    st.image(client.job_photo_url(job, pid), use_container_width=True)
                    if st.button(f"{side}: {pid}", key=f"cmp_sw_{side}_{a_pid}_{b_pid}",
                                 use_container_width=True):
                        _open([a_pid, b_pid])


def _people(client, cmp: Dict) -> None:
    st.caption("Largest first. People are numbered per run, so the same person can have a different "
               "number in A and B; compare by face and size.")
    a_people, b_people = cmp["people"][cmp["a"]], cmp["people"][cmp["b"]]
    n = max(len(a_people), len(b_people))
    limit = n if st.toggle("Show everyone", key="compare_people_all") else min(n, 15)
    head = st.columns([1, 3, 1, 3])
    head[1].markdown(f"**A** ({len(a_people)} people)")
    head[3].markdown(f"**B** ({len(b_people)} people)")
    for i in range(limit):
        cols = st.columns([1, 3, 1, 3])
        for img_col, txt_col, people, job in ((cols[0], cols[1], a_people, cmp["a"]),
                                              (cols[2], cols[3], b_people, cmp["b"])):
            if i >= len(people):
                continue
            person = people[i]
            if person["face_id"] is not None:
                img_col.image(client.job_face_url(job, person["face_id"]), width=64)
            txt_col.write(f"{person['label'] or person['id']}: **{person['face_count']}** faces")


@st.dialog("Photo in A and B", width="large")
def _photo_dialog(client, cmp: Dict, photo_ids: List[str]) -> None:
    st.session_state["compare_photo"] = None          # closing the dialog must not reopen it
    for pid in photo_ids:
        st.markdown(f"**{pid}**")
        img, a_col, b_col = st.columns([2, 2, 2])
        img.image(client.job_photo_url(cmp["b"], pid), use_container_width=True)
        for col, side, job in ((a_col, "A", cmp["a"]), (b_col, "B", cmp["b"])):
            d = cmp["decisions"][job].get(pid, {})
            col.markdown(f"**{side}: {STATE_LABEL.get(d.get('state'), d.get('state', '?'))}**")
            col.write(d.get("detail", ""))
            score = d.get("score")
            col.caption(f"scene {d.get('scene')}" + (f", score {score:.2f}" if score is not None else ""))
