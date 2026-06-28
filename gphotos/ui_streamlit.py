"""Streamlit 'Import from Google Photos' widget (spec-092, P5).

A thin UI adapter over :func:`gphotos.ingest_source.import_from_google_photos`.
Streamlit is imported lazily inside the function so the core ``gphotos`` package
stays framework-agnostic (importable from notebooks / the pipeline). Shared by
both the Albumify album creator and the Face Clustering run tab.

Local-app flow: clicking the button runs the import synchronously -- the OAuth
consent (first time) and the Google picker open in the user's browser, the call
blocks until they finish picking, then the imported directory is written into the
target text field. This is fine because both apps run locally for one user.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path.home() / ".sim_bench" / "gphotos_imports"
_NOTICE_KEY = "_gphotos_notice"


def render_import_button(
    *,
    key: str,
    target_key: str,
    client_secret: str = "client_secret.json",
    cache_root: Path | str | None = None,
    label: str = "Import from Google Photos",
) -> None:
    """Render an import control that fills ``st.session_state[target_key]``.

    Args:
        key: unique widget id + the per-site cache subdir name.
        target_key: session_state key of the directory text field to populate
            (FC: ``"last_image_dir"``; Albumify: ``"new_album_source"``).
        client_secret: path to the Google OAuth client secret JSON.
        cache_root: where downloads land (default ``~/.sim_bench/gphotos_imports``).
        label: button caption.
    """
    import streamlit as st

    # Surface the one-shot notice from a prior import (set before st.rerun()).
    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        st.success(notice)
        st.info(
            "Note: photos imported from Google Photos carry no GPS location "
            "(Google strips it on download), so trip/geo detection will be "
            "limited. Use a Google Takeout folder if you need location."
        )

    if not Path(client_secret).exists():
        st.caption(
            ":grey[Google Photos import unavailable - missing client_secret.json "
            "(see specs/092-google-photos-integration/IMPLEMENTATION_GUIDE.html).]"
        )
        return

    if not st.button(f"\U0001F4F7 {label}", key=f"{key}_gphotos_btn"):
        return

    out_dir = Path(cache_root or DEFAULT_CACHE_ROOT) / key
    with st.spinner(
        "Opening Google Photos in your browser - pick photos, click Done, "
        "then return here..."
    ):
        try:
            from gphotos.ingest_source import import_from_google_photos

            res = import_from_google_photos(client_secret, out_dir)
        except Exception as exc:  # surface, never crash the page
            logger.exception("Google Photos import failed")
            st.error(f"Import failed: {exc}")
            return

    if res.count == 0:
        st.warning("No photos were picked.")
        return

    st.session_state[target_key] = str(res.source_directory)
    st.session_state[_NOTICE_KEY] = (
        f"Imported {res.count} photo(s) to {res.source_directory}"
    )
    st.rerun()
