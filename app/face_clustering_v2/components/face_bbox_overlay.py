"""spec-064 — face bbox + landmarks overlay (Plotly).

Renders the source image with the focus face's bbox rectangle and the
five landmark dots (when available). Falls back to the aligned crop if
the source image is missing on disk.

The component is Streamlit-bound (uses ``st.plotly_chart``); helpers
that compute the figure are kept free of Streamlit so tests can render
without a UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st


def render_face_bbox_overlay(
    *,
    source_image_path: str,
    crop_fallback: Optional[Path],
    bbox: Optional[tuple] = None,
    landmarks=None,
    pose: Optional[tuple] = None,
) -> None:
    """Render the source photo (or crop fallback) with the bbox overlay.

    ``bbox`` is ``(x1, y1, x2, y2)`` in source-image pixel coords (matches
    ``FaceRecord.bbox``). ``pose`` is ``(yaw, pitch, roll)`` degrees (spec-070);
    when present (and a bbox is available) the 3 head-pose axes are drawn
    anchored at the bbox centre (X red, Y green, Z blue). The bbox / landmarks
    / pose parameters only apply to the source-image render, not the crop
    fallback.
    """
    src = Path(source_image_path) if source_image_path else None
    if src and src.is_file():
        try:
            import plotly.graph_objects as go
            from PIL import Image, ImageOps

            # Apply EXIF orientation so phone photos display upright. The stored
            # bbox / landmarks are already in the upright (EXIF-corrected) frame
            # — the detector ran on the oriented image — so without this the
            # image showed sideways while the bbox sat in the wrong place
            # (user report 2026-06-05). exif_transpose also strips the tag, so
            # img.width/height below are the correct upright dimensions.
            img = ImageOps.exif_transpose(Image.open(src))
            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=img, xref="x", yref="y",
                    x=0, y=img.height, sizex=img.width, sizey=img.height,
                    sizing="stretch", layer="below",
                )
            )
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                fig.add_shape(
                    type="rect",
                    x0=float(x1), y0=img.height - float(y1),
                    x1=float(x2), y1=img.height - float(y2),
                    line=dict(color="lime", width=3),
                )
            if landmarks is not None and len(landmarks) > 0:
                xs = [float(p[0]) for p in landmarks]
                ys = [img.height - float(p[1]) for p in landmarks]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=8, color="cyan"), showlegend=False,
                ))
            # spec-070: head-pose axes (X red / Y green / Z blue) anchored at
            # the bbox centre. y is flipped to match the layout image.
            if (pose is not None and len(pose) == 3
                    # ``v == v`` is False for NaN — legacy runs (pre-SIGHTING-093)
                    # store NaN pose, which would draw garbage axes otherwise.
                    and all(v is not None and v == v for v in pose)
                    and bbox is not None and len(bbox) == 4):
                from face_cluster.overlays import pose_axes_2d

                bx1, by1, bx2, by2 = (float(v) for v in bbox)
                center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                scale = 0.5 * min(abs(bx2 - bx1), abs(by2 - by1))
                axes = pose_axes_2d(center, scale, pose[0], pose[1], pose[2])
                colours = {"x": "red", "y": "lime", "z": "deepskyblue"}
                for k, (ax0, ay0, ax1, ay1) in axes.items():
                    fig.add_trace(go.Scatter(
                        x=[ax0, ax1], y=[img.height - ay0, img.height - ay1],
                        mode="lines", line=dict(color=colours[k], width=4),
                        name=f"{k}-axis", showlegend=False,
                    ))
            # Preserve aspect ratio (user report 2026-06-05: image was stretched).
            # Scale the figure to the image's proportions, capped to a max edge,
            # and lock 1 x-unit == 1 y-unit so the photo can't distort.
            max_edge = 720
            s = min(1.0, max_edge / max(img.width, img.height))
            disp_w, disp_h = int(img.width * s), int(img.height * s)
            fig.update_xaxes(visible=False, range=[0, img.width])
            fig.update_yaxes(visible=False, range=[0, img.height],
                             scaleanchor="x", scaleratio=1)
            fig.update_layout(
                width=disp_w, height=disp_h, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#111", plot_bgcolor="#111",
            )
            st.plotly_chart(fig, use_container_width=False)
            return
        except Exception:  # noqa: BLE001
            # Fall through to crop fallback on any rendering failure.
            pass

    if crop_fallback is not None and crop_fallback.is_file():
        try:
            st.image(str(crop_fallback), width=400)
        except Exception:  # noqa: BLE001
            st.caption("(crop unreadable)")
    else:
        st.caption("(no image available)")
