"""Occlusion Review app (spec-096) — adjudication + explainability.

Pages: Adjudicate (resolve Haiku-vs-user disagreements incl. the
foreground-object category) - LoG Explain (patch-grid overlay + LR
contributions) - Model Table - Error Gallery.

Run:  .venv/Scripts/streamlit run app/occlusion_review/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import streamlit as st

try:  # SIGHTING-114: 124/776 negatives are HEIC; PIL needs the opener registered
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from app.occlusion_review import data as D
from app.occlusion_review import explain as E

st.set_page_config(page_title="Occlusion Review", layout="wide")
ROOT = D.dataset_root()


@st.cache_data(show_spinner="Loading dataset + model opinions...")
def _records():
    return D.load_all(ROOT)


@st.cache_resource(show_spinner="Fitting LoG explainer...")
def _explainer():
    return E.LogExplainer(ROOT)


def _opinions(rec):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("user label", "occluded" if rec["label"] == "1" else "clean")
    hk = "occluded" if rec["haiku_occluded"] == "1" else ("clean" if rec["haiku_occluded"] == "0" else "?")
    c2.metric("Haiku", f"{hk} L{rec['haiku_level'] or '-'}")
    c3.metric("classical", f"{rec['classical_score']:.2f}")
    lp = rec["log_prob"]
    c4.metric("LoG P(occl)", "-" if lp != lp else f"{lp:.2f}")
    bp = rec.get("B_clip_oof", float("nan"))
    c5.metric("CLIP probe (OOF)", "-" if bp != bp else f"{bp:.2f}")
    if rec["haiku_reason"]:
        st.caption(f"Haiku: {rec['haiku_reason']}")


def page_adjudicate(records):
    st.subheader("Adjudicate disagreements")
    corr = D.load_corrections(ROOT)
    work = D.disagreements(records)
    todo = [r for r in work if r["id"] not in corr]
    st.progress(1 - len(todo) / max(len(work), 1),
                text=f"{len(work) - len(todo)} / {len(work)} reviewed")
    if not todo:
        st.success("All disagreements adjudicated. corrections.csv is complete.")
        return
    rec = todo[0]
    mine = "OCCLUDED" if rec["label"] == "1" else "CLEAN"

    def _save(decision: str) -> None:
        notes = st.session_state.get(f"n_{rec['id']}", "")
        D.save_correction(rec["id"], rec["label"], decision, notes, ROOT)
        st.rerun()

    # One-click row, ABOVE the image so no scrolling. Primary = confirm the
    # user's original label; every button saves immediately and advances.
    b = st.columns([2.4, 1.7, 1.9, 0.9, 0.9, 0.9])
    if b[0].button(f"CONFIRM MINE: {mine}", type="primary", use_container_width=True,
                   help="Keep your original label and move to the next image"):
        _save("occluded" if mine == "OCCLUDED" else "clean")
    other = "clean" if mine == "OCCLUDED" else "occluded"
    if b[1].button(f"no - {other}", use_container_width=True,
                   help="Haiku was right; flip the label"):
        _save(other)
    if b[2].button("foreground object", use_container_width=True,
                   help="branch/head/strap near the camera, NOT on the lens"):
        _save("foreground_object")
    for i, lv in enumerate(("L1", "L2", "L3")):
        if b[3 + i].button(lv, use_container_width=True,
                           help=f"occluded, severity {lv} (optional - only if you want to grade it)"):
            _save(f"occluded_{lv.lower()}")

    st.markdown(f"**{rec['id']}** &nbsp; ({rec['source_dataset']}, "
                f"{'flagged by Haiku' if rec['label'] == '0' else 'MISSED by Haiku'})")
    _opinions(rec)
    try:  # SIGHTING-114: never let one unreadable file block the queue
        st.image(D.image_path(ROOT, rec), width=760)
    except Exception as exc:  # buttons above stay usable; adjudicate from the path
        st.error(f"Cannot render {rec['id']}: {exc}. File: {D.image_path(ROOT, rec)}")
    st.text_input("notes (optional, saved with the NEXT button you click)", key=f"n_{rec['id']}")


def page_explain(records):
    st.subheader("LoG Explain — where the detector looked and why")
    ids = [r["id"] for r in records]
    default = st.session_state.get("explain_id", ids[0])
    rid = st.selectbox("image", ids, index=ids.index(default) if default in ids else 0)
    rec = next(r for r in records if r["id"] == rid)
    stat = st.selectbox("patch statistic (heatmap)",
                        ["near_zero_fraction", "tail_ratio", "iqr", "std", "entropy", "p90"])
    _opinions(rec)
    col1, col2 = st.columns([3, 2])
    with col1:
        img = E.overlay(D.image_path(ROOT, rec), stat)
        if img is not None:
            st.image(img, caption=f"heatmap: {stat} per patch - yellow: max cell, cyan: max BORDER cell")
    with col2:
        out = _explainer().explain(D.image_path(ROOT, rec))
        if out:
            prob, rows = out
            st.metric("LoG-LR P(occluded)", f"{prob:.3f}")
            st.caption("Top feature contributions (coef x z-value; + pushes OCCLUDED)")
            st.dataframe(rows, use_container_width=True, height=380)


def page_table(records):
    st.subheader("Model table — every opinion, side by side")
    corr = D.load_corrections(ROOT)
    def _r(v, nd=3):
        return round(v, nd) if v == v else None  # NaN -> None
    rows = [{"id": r["id"], "source": r["source_dataset"], "user": r["label"],
             "corrected": D.effective_label(r, corr) if r["id"] in corr else "",
             "B_clip": _r(r["B_clip_oof"]), "F_log": _r(r["F_log_oof"]),
             "D2_cnn": _r(r["D2_cnn"]), "C_vlm_L": _r(r["C_tinyvlm_level"], 0),
             "haiku": r["haiku_occluded"], "haiku_L": r["haiku_level"],
             "classical": round(r["classical_score"], 3),
             "disagree": (r["haiku_occluded"] != "" and r["haiku_occluded"] != r["label"]),
             "haiku_reason": r["haiku_reason"]} for r in records]
    only_dis = st.checkbox("disagreements only", value=True)
    if only_dis:
        rows = [r for r in rows if r["disagree"]]
    st.dataframe(rows, use_container_width=True, height=560)
    st.caption(f"{len(rows)} rows. Pick an id on the LoG Explain page for the overlay view.")


# model key -> (record field, default threshold, is_binary)
_GALLERY_MODELS = {
    "CLIP probe (B, OOF)": ("B_clip_oof", 0.5, False),
    "LoG stats (F, OOF)": ("F_log_oof", 0.5, False),
    "CNN (D2, synth-trained)": ("D2_cnn", 0.5, False),
    "classical (E)": ("classical_score", 0.001, False),
    "tiny VLM (C, level)": ("C_tinyvlm_level", 0.5, False),
    "Haiku (A)": ("haiku_occluded", 0.5, True),
}


def page_gallery(records):
    st.subheader("Error gallery — per-model misses & false alarms")
    corr = D.load_corrections(ROOT)
    c1, c2, c3 = st.columns([2, 2, 1.4])
    model = c1.selectbox("model", list(_GALLERY_MODELS))
    kind = c2.radio("error type", ["misses (occluded, model says clean)",
                                   "false alarms (clean, model flags)"], horizontal=False)
    field, thr_default, is_binary = _GALLERY_MODELS[model]
    thr = thr_default if is_binary else c3.slider("threshold", 0.0, 1.0, float(thr_default), 0.05)

    def score(r):
        v = r.get(field)
        if is_binary:
            return 1.0 if v == "1" else (0.0 if v == "0" else float("nan"))
        try:
            f = float(v)
            return f if f == f else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    def lbl(r):
        return D.effective_label(r, corr)

    fg_skipped = sum(1 for r in records if lbl(r) == "fg")
    scored = [(r, score(r)) for r in records if lbl(r) in ("0", "1")]
    scored = [(r, s) for r, s in scored if s == s]  # drop unscored (e.g. VLM in progress)
    if "misses" in kind:
        sel = [(r, s) for r, s in scored if lbl(r) == "1" and s < thr]
        sel.sort(key=lambda t: t[1])          # most confidently wrong first
    else:
        sel = [(r, s) for r, s in scored if lbl(r) == "0" and s >= thr]
        sel.sort(key=lambda t: -t[1])
    st.caption(f"{len(sel)} errors of {sum(1 for r,_ in scored if lbl(r)== ('1' if 'misses' in kind else '0'))} "
               f"eligible images - labels include your corrections"
               + (f" - {fg_skipped} foreground-object images excluded" if fg_skipped else "")
               + " - sorted most-confidently-wrong first")
    cols = st.columns(4)
    for i, (r, s) in enumerate(sel[:40]):
        with cols[i % 4]:
            st.image(D.image_path(ROOT, r), caption=f"{s:.2f} - {r['id'][:30]}",
                     use_container_width=True)
            if st.button("explain", key=f"g_{r['id']}"):
                st.session_state["explain_id"] = r["id"]
                st.session_state["page"] = "LoG Explain"
                st.rerun()


st.title("Occlusion Review")
page = st.sidebar.radio("Page", ["Adjudicate", "LoG Explain", "Model Table", "Error Gallery"],
                        key="page")
recs = _records()
st.sidebar.caption(f"dataset: {ROOT}  -  {len(recs)} images")
{"Adjudicate": page_adjudicate, "LoG Explain": page_explain,
 "Model Table": page_table, "Error Gallery": page_gallery}[page](recs)
