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
    st.markdown(f"**{rec['id']}** &nbsp; ({rec['source_dataset']}, "
                f"{'flagged by Haiku' if rec['label'] == '0' else 'MISSED by Haiku'})")
    _opinions(rec)
    st.image(D.image_path(ROOT, rec), width=760)
    choice = st.radio("Your verdict", ["occluded L1 (slight)", "occluded L2 (moderate)",
                                       "occluded L3 (severe)", "clean",
                                       "foreground object (branch/head/strap - not on lens)"],
                      horizontal=False, key=f"v_{rec['id']}")
    notes = st.text_input("notes (optional)", key=f"n_{rec['id']}")
    if st.button("Save & next", type="primary"):
        decision = {"occluded L1 (slight)": "occluded_l1", "occluded L2 (moderate)": "occluded_l2",
                    "occluded L3 (severe)": "occluded_l3", "clean": "clean",
                    "foreground object (branch/head/strap - not on lens)": "foreground_object"}[choice]
        D.save_correction(rec["id"], rec["label"], decision, notes, ROOT)
        st.rerun()


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


def page_gallery(records):
    st.subheader("Error gallery — pattern spotting")
    corr = D.load_corrections(ROOT)
    mode = st.selectbox("show", [
        "Haiku misses (label=occluded, haiku=clean)",
        "Haiku false alarms (label=clean, haiku=occluded)",
        "classical misses (score=0 on occluded)",
        "LoG misses (P<0.5 on occluded)",
    ])
    def lbl(r):
        return D.effective_label(r, corr)
    if "Haiku misses" in mode:
        sel = [r for r in records if lbl(r) == "1" and r["haiku_occluded"] == "0"]
    elif "false alarms" in mode:
        sel = [r for r in records if lbl(r) == "0" and r["haiku_occluded"] == "1"]
    elif "classical" in mode:
        sel = [r for r in records if lbl(r) == "1" and r["classical_score"] == 0.0]
    else:
        sel = [r for r in records if lbl(r) == "1" and r["log_prob"] == r["log_prob"] and r["log_prob"] < 0.5]
    st.caption(f"{len(sel)} images (labels include your corrections)")
    cols = st.columns(4)
    for i, r in enumerate(sel[:40]):
        with cols[i % 4]:
            st.image(D.image_path(ROOT, r), caption=f"{r['id'][:34]}", use_container_width=True)
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
