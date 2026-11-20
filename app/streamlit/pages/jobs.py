"""Jobs page (spec-112 TA5.3, spec-115 W4, spec-116 R15): examine jobs. Runs start on Configure & Run only.

Watch a job live and cancel it; each job links to its run in Albums. A finished job shows its picks, every dropped photo with its reason, photos a filter removed
before the album saw them, and people.
Promote / demote / rename write user edits through the API; the album is rebuilt from the rows on
every view. Clickable actions are real st.button grids (never dataframe row-select).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

import streamlit as st

from app.streamlit.api_client import ApiError, get_client

logger = logging.getLogger(__name__)

COLS = 6
PAGE = 48
REASON_LABEL = {"near_dup": "Near-duplicate", "not_selected": "Not selected",
                "set_aside": "Set aside (triage)"}
RUNNER_LABEL = {"local": "This PC", "modal_gpu": "Modal L4 (GPU)", "modal_cpu": "Modal CPU (slow, free tier)"}
TERMINAL = ("succeeded", "failed")


def render_jobs_page() -> None:
    st.title("Jobs")
    client = get_client()
    try:
        jobs = client.list_jobs()
    except ApiError as e:
        st.error(f"Cannot list jobs: {e}")
        return
    if not jobs:
        st.info("No jobs yet. Start a run from Configure & Run.")
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


# --- watch and cancel (spec-115 W4) ----------------------------------------------------------------

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


def _render_job_tools(client, job: Dict) -> None:
    try:
        client.get_pipeline_status(job["id"])
    except ApiError:
        st.caption("A test or script run: it has no row in Albums.")
    else:
        if st.button("Open this run in Albums", key=f"jobs_open_{job['id']}"):
            st.session_state["albums_focus_run"] = job["id"]
            st.session_state["current_page"] = "albums"
            st.rerun()
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
