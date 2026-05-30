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
) -> None:
    """Render the source photo (or crop fallback) with the bbox overlay.

    ``bbox`` is ``(x1, y1, x2, y2)`` in source-image pixel coords (matches
    ``FaceRecord.bbox``). The bbox / landmarks parameters are not used for
    the fallback crop render — they only make sense over the source image.
    """
    src = Path(source_image_path) if source_image_path else None
    if src and src.is_file():
        try:
            import plotly.graph_objects as go
            from PIL import Image

            img = Image.open(src)
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
            fig.update_xaxes(visible=False, range=[0, img.width])
            fig.update_yaxes(visible=False, range=[0, img.height])
            fig.update_layout(
                height=480, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#111", plot_bgcolor="#111",
            )
            st.plotly_chart(fig, use_container_width=True)
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
