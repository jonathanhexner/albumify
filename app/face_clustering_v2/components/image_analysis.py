"""spec-081 — Image Analysis panel.

Given an ``ImageDetail`` (RunStore), renders the source photo with EVERY face's
bbox overlaid (colour-coded by disposition), a per-face table, and the
image-level scores. Read-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# disposition -> bbox colour
_COLOUR = {"clustered": "#22c55e", "noise": "#f59e0b", "filtered": "#ef4444"}


def _disposition(face: Any) -> str:
    if face.cluster_id is not None and face.cluster_id >= 0:
        return "clustered"
    if face.rejection_reason:
        return "filtered"
    return "noise"


def faces_to_box(faces, *, show_filtered: bool):
    """spec-083: faces that get a bbox drawn — valid bbox, and (passed filtration
    OR the user opted to also see rejected). Pure; unit-tested."""
    out = []
    for f in faces:
        if not f.bbox or len(f.bbox) != 4:
            continue
        if _disposition(f) == "filtered" and not show_filtered:
            continue
        out.append(f)
    return out


def render_image_analysis(detail: Any) -> None:
    """Source image + face boxes (passing by default) + identities + per-face table."""
    src = Path(detail.image_path)
    n = len(detail.faces)
    counts = {d: sum(1 for f in detail.faces if _disposition(f) == d)
              for d in ("clustered", "noise", "filtered")}
    # identities present = clusters of the clustered faces (the point of the run)
    identities = sorted({f.cluster_id for f in detail.faces
                         if f.cluster_id is not None and f.cluster_id >= 0})
    st.markdown(
        f"**{src.name}** — {n} faces · "
        f":green[{counts['clustered']} clustered] · "
        f":orange[{counts['noise']} noise] · :red[{counts['filtered']} filtered]"
    )
    who = ", ".join(f"C{c}" for c in identities) if identities else "none"
    st.caption(f"People in this photo: **{who}**  ·  boxes show faces that passed "
               "filtration (green=clustered, amber=noise).")

    # spec-083: by default draw only the faces that PASSED filtration (the user's
    # ask); reveal the rejected ones on demand.
    show_filtered = st.checkbox(
        f"Show rejected (filtered) faces too — {counts['filtered']} hidden",
        value=False, key="v2_ia_show_filtered", disabled=counts["filtered"] == 0,
    )
    if src.is_file():
        _render_overlay(detail, src, show_filtered=show_filtered)
    else:
        st.caption(f"Source image not on disk: {detail.image_path}")

    # image-level scores
    scores = {k: getattr(detail, k, None) for k in ("iqa_score", "ava_score", "sharpness_score")}
    st.caption("Image scores: " + " · ".join(
        f"{k.replace('_score','').upper()}={'-' if v is None else f'{v:.3f}'}" for k, v in scores.items()
    ))

    # per-face table
    rows = []
    for f in detail.faces:
        rows.append({
            "face": f.face_id,
            "disposition": _disposition(f),
            "cluster": ("C" + str(f.cluster_id)) if (f.cluster_id is not None and f.cluster_id >= 0) else "-",
            "gate": f.rejection_reason or "passed",
            "blur": round(f.blur_score, 0),
            "area": round(f.area, 0),
            "det": None if f.det_score is None else round(f.det_score, 3),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_overlay(detail: Any, src: Path, *, show_filtered: bool = False) -> None:
    import plotly.graph_objects as go
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(src))  # upright; bboxes are in this frame
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=img, xref="x", yref="y", x=0, y=img.height,
        sizex=img.width, sizey=img.height, sizing="stretch", layer="below",
    ))
    for f in faces_to_box(detail.faces, show_filtered=show_filtered):
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        colour = _COLOUR[_disposition(f)]
        fig.add_shape(type="rect", x0=x1, y0=img.height - y1, x1=x2, y1=img.height - y2,
                      line=dict(color=colour, width=3))
        fig.add_annotation(x=x1, y=img.height - y1, text=str(f.face_id), showarrow=False,
                           font=dict(color="white", size=11), xanchor="left", yanchor="bottom",
                           bgcolor=colour, opacity=0.85)
    max_edge = 820
    s = min(1.0, max_edge / max(img.width, img.height))
    fig.update_xaxes(visible=False, range=[0, img.width])
    fig.update_yaxes(visible=False, range=[0, img.height], scaleanchor="x", scaleratio=1)
    fig.update_layout(width=int(img.width * s), height=int(img.height * s),
                      margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#111", plot_bgcolor="#111")
    st.plotly_chart(fig, use_container_width=False)
