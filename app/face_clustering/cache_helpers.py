"""Caching helpers for loading manifests, face DataFrames, and crops."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image


def _load_manifest(output_dir: Path) -> dict:
    cached = st.session_state.manifest_cache
    if cached and cached[0] == str(output_dir):
        return cached[1]
    path = output_dir / "crop_manifest.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    st.session_state.manifest_cache = (str(output_dir), m)
    return m


def _load_faces_df(output_dir: Path) -> pd.DataFrame:
    cached = st.session_state.faces_df_cache
    if cached and cached[0] == str(output_dir):
        return cached[1]
    df = pd.read_csv(output_dir / "faces.csv")
    st.session_state.faces_df_cache = (str(output_dir), df)
    return df


def _crop_for_face(face_id: int, output_dir: Path) -> Optional[Image.Image]:
    manifest = _load_manifest(output_dir)
    rel = manifest.get(str(face_id))
    if not rel:
        return None
    p = Path(rel) if Path(rel).is_absolute() else output_dir / rel
    return Image.open(p) if p.exists() else None
