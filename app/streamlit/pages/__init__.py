"""Streamlit pages package."""

from .home import render_home_page
from .albums import render_albums_page
from .results import render_results_page
from .people import render_people_page

__all__ = [
    "render_home_page",
    "render_albums_page",
    "render_results_page",
    "render_people_page",
]
