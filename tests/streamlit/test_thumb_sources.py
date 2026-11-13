"""SIGHTING-152 C5: thumbnails are made from the run's 768 px copy whenever the API named one."""
from pathlib import Path

from PIL import Image

from app.streamlit import thumb_sources
from app.streamlit.api_client import ApiClient


def test_a_parsed_image_points_every_thumbnail_loader_at_its_copy(tmp_path):
    from app.streamlit.components.gallery import _load_thumbnail
    from app.streamlit.pages.explore import _load_thumb
    from app.streamlit.components.metrics import _build_metric_row
    from app.streamlit.pages.results import _load_thumbnail as results_thumb

    copy = tmp_path / "copy.jpg"
    Image.new("RGB", (768, 512), "red").save(copy)
    original = r"D:\gone\IMG_1.jpg"        # not on this machine: only the copy can make the thumbnail
    image = ApiClient.__new__(ApiClient)._parse_image({"path": original, "display_path": str(copy)})

    assert image.display_path == str(copy)
    assert thumb_sources.source("D:/gone/IMG_1.jpg") == str(copy)      # either slash spelling
    assert _load_thumbnail(Path(original)).size == (200, 133)
    assert _load_thumb(original).startswith("data:image/jpeg;base64,")
    assert results_thumb(original).startswith("data:image/jpeg;base64,")
    assert _build_metric_row(image, is_sel=True)["Thumbnail"].startswith("data:image/jpeg;base64,")


def test_a_path_without_a_copy_is_read_as_is(tmp_path):
    thumb_sources.remember(str(tmp_path / "a.jpg"), None)
    assert thumb_sources.source(str(tmp_path / "a.jpg")) == str(tmp_path / "a.jpg")
