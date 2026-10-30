"""Jobs page (spec-112 TA5.3, spec-115 W4): run albums and examine them.

Run an album (folder, photo range, runner, pipeline; the L4 asks for a confirm with its estimate), watch it
live, cancel it, re-run its photos with another runner or pipeline (Compare runs opens on the pair when it
finishes). A finished job shows its picks, every dropped photo with its reason, photos a filter removed
before the album saw them, and people.
Promote / demote / rename write user edits through the API; the album is rebuilt from the rows on
every view. Clickable actions are real st.button grids (never dataframe row-select).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import yaml

from app.streamlit.api_client import ApiError, get_client

logger = logging.getLogger(__name__)

COLS = 6
PAGE = 48
REASON_LABEL = {"near_dup": "Near-duplicate", "not_selected": "Not selected",
                "set_aside": "Set aside (triage)"}
RUNNER_LABEL = {"local": "This PC", "modal_gpu": "Modal L4 (GPU)", "modal_cpu": "Modal CPU (slow, free tier)"}
TERMINAL = ("succeeded", "failed")
_PIPELINES = Path(__file__).resolve().parents[3] / "configs" / "pipeline.yaml"


def _pipelines() -> List[str]:
    try:
        doc = yaml.safe_load(_PIPELINES.read_text(encoding="utf-8")) or {}
    except OSError:
        return ["default"]
    names = [k[:-len("_pipeline")] for k in doc if k.endswith("_pipeline")]
    return sorted(names, key=lambda n: (n != "default", n)) or ["default"]


def render_jobs_page() -> None:
    st.title("Jobs")
    client = get_client()
    try:
        jobs = client.list_jobs()
    except ApiError as e:
        st.error(f"Cannot list jobs: {e}")
        return
    with st.expander("Run an album", expanded=not jobs):
        _render_run_form(client)
    if not jobs:
        st.info("No jobs yet. Run an album above.")
        return

    if not st.toggle("Show regression runs", key="jobs_show_regression",
                     help="Runs of scripts/albumify_regression.py (tag regression:<album>). "
                          "Compare them on the Compare runs page."):
        jobs = [j for j in jobs if not (j.get("tag") or "").startswith("regression:")]
        if not jobs:
            st.info("No albums of your own yet; switch on 'Show regression runs' to see the test runs.")
            return
    labels = {j["id"]: f"{j['id']}  -  {j['status']}  ({j['runner']}, {j['n_inputs']} photos)"
                       + (f"  [{j['tag']}]" if j.get("tag") else "")
              for j in jobs}
    if st.session_state.get("jobs_select_next") in labels:
        st.session_state["jobs_selected"] = st.session_state.pop("jobs_select_next")
    job_id = st.selectbox("Job", list(labels), format_func=labels.get, key="jobs_selected")
    job = client.get_job(job_id)
    if job["status"] not in TERMINAL:
        _live_status(job_id)
        return
    _render_status(job)
    _render_job_tools(client, job)
    if job["status"] != "succeeded":
        return

    album = client.get_job_album(job_id)
    photos = client.list_job_photos(job_id)
    _render_header(album, photos)
    removed = [p for p in photos if not p["in_album"]]
    tabs = st.tabs([f"Picks ({len(album['picks'])})", f"Dropped ({len(album['dropped'])})",
                    f"Removed by filters ({len(removed)})", f"People ({len(album['people'])})"])
    with tabs[0]:
        _render_picks(client, job_id, album)
    with tabs[1]:
        _render_dropped(client, job_id, album)
    with tabs[2]:
        _render_removed(client, job_id, removed)
    with tabs[3]:
        _render_people(client, job_id, album)


def _render_status(job: Dict) -> None:
    status = job["status"]
    if status == "failed":
        st.error(f"Failed{' at ' + job['failed_step'] if job.get('failed_step') else ''}: "
                 f"{job.get('error')}")
    with st.expander("Job history and settings"):
        settings = job.get("settings") or {}
        if settings:
            st.caption(f"Pipeline {settings.get('pipeline')} - {RUNNER_LABEL.get(settings.get('runner'), settings.get('runner'))}"
                       f" ({settings.get('device')}) - code {settings.get('commit') or 'unknown'} - "
                       f"{len(settings.get('steps') or [])} steps")
        for e in job.get("events", []):
            st.text(f"{e['ts'][:19]}  {e['kind']:<8} {e['message']}")


# --- run, watch, cancel, re-run (spec-115 W4) ------------------------------------------------------

def _start_controls(client, key: str, *, source: Optional[str] = None, start: int = 0,
                    n: Optional[int] = None, rerun_of: Optional[str] = None) -> None:
    """Runner + pipeline + Start. The L4 shows its estimate and needs a second click (R2)."""
    c1, c2 = st.columns(2)
    runner = c1.selectbox("Run on", list(RUNNER_LABEL), format_func=RUNNER_LABEL.get, key=f"{key}_runner")
    pipeline = c2.selectbox("Pipeline", _pipelines(), key=f"{key}_pipeline")
    pending = st.session_state.get(f"{key}_pending")
    args = dict(source=source, start=start, n=n, runner=runner, rerun_of=rerun_of)
    if pending and pending != {**args, "pipeline": pipeline}:
        pending = st.session_state[f"{key}_pending"] = None      # a field changed: estimate again
    if not pending:
        if st.button("Start", key=f"{key}_go", type="primary", disabled=not (source or rerun_of)):
            try:
                est = client.estimate_job(**args)
            except ApiError as e:
                st.error(str(e))
                return
            if not est["allowed"]:
                st.error(est["reason"])
                return
            if runner == "modal_gpu":
                st.session_state[f"{key}_pending"] = {**args, "pipeline": pipeline}
                st.session_state[f"{key}_estimate"] = est
                st.rerun()
            _submit(client, key, {**args, "pipeline": pipeline})
        return
    est = st.session_state.get(f"{key}_estimate") or {}
    st.warning(f"This runs on the Modal L4: {est.get('n_photos')} photos, about {est.get('gpu_minutes')} GPU "
               f"minutes, about ${est.get('dollars')}. Photos Modal already has are not uploaded again.")
    b1, b2 = st.columns(2)
    if b1.button("Confirm: start on the L4", key=f"{key}_confirm", type="primary"):
        st.session_state[f"{key}_pending"] = None
        _submit(client, key, pending)
    if b2.button("Back", key=f"{key}_back"):
        st.session_state[f"{key}_pending"] = None
        st.rerun()


def _submit(client, key: str, body: Dict) -> None:
    try:
        job = client.submit_job(**body)
    except ApiError as e:
        st.error(str(e))
        return
    if body.get("rerun_of"):
        st.session_state["jobs_rerun_pairs"] = {**st.session_state.get("jobs_rerun_pairs", {}),
                                                job["job_id"]: body["rerun_of"]}
    # the Job selector may already be drawn in this run (re-run tools sit below it): select on the next run
    st.session_state["jobs_select_next"] = job["job_id"]
    st.rerun()


def _render_run_form(client) -> None:
    source = st.text_input("Photo folder on this PC", key="jobs_run_source", placeholder="D:/Slovakia26")
    c1, c2 = st.columns(2)
    start = int(c1.number_input("First photo (0 = from the start)", min_value=0, value=0, key="jobs_run_start"))
    n = int(c2.number_input("How many (0 = all)", min_value=0, value=0, key="jobs_run_n"))
    _start_controls(client, "jobs_run", source=source.strip() or None, start=start, n=n or None)


@st.fragment(run_every=2)
def _live_status(job_id: str) -> None:
    """Refreshes itself until the job ends (R3); then the whole page reloads to show the album."""
    client = get_client()
    try:
        job = client.get_job(job_id)
    except ApiError as e:
        st.error(str(e))
        return
    if job["status"] in TERMINAL:
        pairs = st.session_state.get("jobs_rerun_pairs", {})
        if job["status"] == "succeeded" and job_id in pairs:
            _open_compare(pairs.pop(job_id), job_id)       # R5: Compare runs opens on the pair
        st.rerun()
    started = job.get("started_at") or job["created_at"]
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(started.replace("Z", "+00:00")).replace(
        tzinfo=timezone.utc)
    last = (job.get("events") or [{}])[-1]
    st.info(f"**{job['status'].capitalize()}** on {RUNNER_LABEL.get(job['runner'], job['runner'])} - "
            f"{int(elapsed.total_seconds() // 60)} min {int(elapsed.total_seconds() % 60)} s")
    if last:
        st.caption(f"Latest: {last.get('message', '')}")
    if st.button("Cancel this job", key=f"jobs_cancel_{job_id}"):
        try:
            client.cancel_job(job_id)
        except ApiError as e:
            st.error(str(e))


def _open_compare(reference: str, new: str) -> None:
    from app.streamlit.pages.compare import ANY
    st.session_state["compare_dataset"] = ANY
    st.session_state[f"compare_a_{ANY}"] = reference
    st.session_state[f"compare_b_{ANY}"] = new
    st.session_state["current_page"] = "compare"


def _render_job_tools(client, job: Dict) -> None:
    with st.expander("Re-run these photos"):
        st.caption("The same photos, with another runner or pipeline. Compare runs opens on the pair when it finishes.")
        _start_controls(client, f"jobs_rerun_{job['id']}", rerun_of=job["id"])
    if job["runner"].startswith("modal"):
        with st.expander("Photos on Modal"):
            st.caption("Modal deletes photos no job used for 7 days. Delete this job's photos now "
                       "(photos another job used in the last 7 days stay).")
            if st.button("Delete this job's photos from Modal", key=f"jobs_del_{job['id']}"):
                try:
                    res = client.delete_remote_inputs(job["id"])
                    st.success(f"Deleted {res['deleted']} photos; kept {res['kept']} used by another job.")
                except ApiError as e:
                    st.error(str(e))


def _render_header(album: Dict, photos: List[Dict]) -> None:
    c = album["counts"]
    cols = st.columns(6)
    for col, (label, value) in zip(cols, [("Photos", len(photos)), ("In album", c["uploaded"]),
                                          ("Picks", c["kept"]), ("Scenes", c["scenes"]),
                                          ("People", c["people"]), ("Set aside", c["set_aside"])]):
        col.metric(label, value)
    theme = album.get("theme") or {}
    if theme.get("label") or theme.get("summary"):
        st.caption(f"Theme: {theme.get('label') or ''} {theme.get('summary') or ''}".strip())


def _edit(client, job_id: str, kind: str, payload: Dict) -> None:
    try:
        client.add_job_edit(job_id, kind, payload)
    except ApiError as e:
        st.error(str(e))
        return
    st.rerun()


def _paged(items: List, key: str) -> List:
    if len(items) <= PAGE:
        return items
    n_pages = (len(items) + PAGE - 1) // PAGE
    page = st.number_input(f"Page (1-{n_pages})", 1, n_pages, 1, key=key)
    return items[(page - 1) * PAGE: page * PAGE]


def _render_picks(client, job_id: str, album: Dict) -> None:
    picks = _paged(album["picks"], f"picks_page_{job_id}")
    for row in range(0, len(picks), COLS):
        cols = st.columns(COLS)
        for col, p in zip(cols, picks[row:row + COLS]):
            with col:
                st.image(client.job_photo_url(job_id, p["id"]), use_container_width=True)
                s = p["score"]
                st.caption(f"#{p['order'] + 1} {p['id']}  scene {p['scene_cluster']}  "
                           f"score {s['composite']:.2f}" if s.get("composite") is not None
                           else f"#{p['order'] + 1} {p['id']}")
                if p["beat"]:
                    st.caption(f"beat {len(p['beat'])} near-duplicate(s)")
                if st.button("Remove", key=f"demote_{job_id}_{p['id']}", use_container_width=True):
                    _edit(client, job_id, "demote", {"photo_id": p["id"]})


def _render_dropped(client, job_id: str, album: Dict) -> None:
    by_reason: Dict[str, List[Dict]] = {}
    for d in album["dropped"]:
        by_reason.setdefault(d["reason"], []).append(d)
    for reason, items in sorted(by_reason.items()):
        with st.expander(f"{REASON_LABEL.get(reason, reason)} ({len(items)})",
                         expanded=len(by_reason) == 1):
            shown = _paged(items, f"dropped_page_{job_id}_{reason}")
            for row in range(0, len(shown), COLS):
                cols = st.columns(COLS)
                for col, d in zip(cols, shown[row:row + COLS]):
                    with col:
                        st.image(client.job_photo_url(job_id, d["id"]), use_container_width=True)
                        detail = d["reason_detail"]
                        if d["beaten_by"]:
                            detail += f" ({', '.join(d['beaten_by'])})"
                        st.caption(f"{d['id']}: {detail}")
                        if d["promotable"] and st.button("Add to album",
                                                         key=f"promote_{job_id}_{d['id']}",
                                                         use_container_width=True):
                            _edit(client, job_id, "promote", {"photo_id": d["id"]})


def _render_removed(client, job_id: str, removed: List[Dict]) -> None:
    if not removed:
        st.success("Every photo reached the album.")
        return
    st.caption("Scored, then removed by a filter (e.g. the sharpness gate) before scene "
               "clustering, so they are not in the album (SIGHTING-128). The filter's own reason "
               "is shown.")
    shown = _paged(removed, f"removed_page_{job_id}")
    for row in range(0, len(shown), COLS):
        cols = st.columns(COLS)
        for col, p in zip(cols, shown[row:row + COLS]):
            with col:
                st.image(client.job_photo_url(job_id, p["photo_id"]), use_container_width=True)
                why = "; ".join(f"{d['step']}: {d['reason']}" for d in p["decisions"]
                                if d.get("decision") == "rejected") or "no recorded reason"
                st.caption(f"{p['photo_id']}: {why}")


def _render_people(client, job_id: str, album: Dict) -> None:
    # largest first: the people the owner most wants to name are the ones with the most faces
    for person in sorted(album["people"], key=lambda p: (-p["face_count"], p["id"])):
        cluster = person["id"].rsplit("_", 1)[-1]
        name = person["label"] or person["id"]
        with st.expander(f"{name}  ({person['face_count']} faces)"):
            faces = client.list_job_faces(job_id, int(cluster)) if cluster.lstrip("-").isdigit() else []
            cols = st.columns(10)
            for col, f in zip(cols, [f for f in faces if f["has_crop"]][:10]):
                col.image(client.job_face_url(job_id, f["face_id"]), width=72)
            c1, c2 = st.columns([3, 1])
            label = c1.text_input("Name", value=person["label"], key=f"name_{job_id}_{person['id']}",
                                  label_visibility="collapsed", placeholder="Name this person")
            if c2.button("Save name", key=f"rename_{job_id}_{person['id']}"):
                _edit(client, job_id, "rename", {"person_id": person["id"], "label": label})
