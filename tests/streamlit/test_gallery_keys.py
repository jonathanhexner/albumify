"""SIGHTING-151: every image card's Detail button gets a unique key, even when two paths share a short hash."""
from streamlit.testing.v1 import AppTest


def _colliding_paths() -> list:
    """Two photo paths whose ``hash(path) % 100000`` match in this process (the old key scheme)."""
    seen = {}
    i = 0
    while True:
        path = f"D:/Trip/IMG_{i:05d}.jpg"
        short = hash(path) % 100000
        if short in seen:
            return [seen[short], path]
        seen[short] = path
        i += 1


def _gallery_app(paths):
    from app.streamlit.components.gallery import render_image_gallery
    from app.streamlit.models import ImageInfo
    render_image_gallery([ImageInfo(path=p, filename=p.rsplit("/", 1)[-1]) for p in paths], columns=4)


def test_colliding_short_hashes_render_without_duplicate_key_error():
    paths = _colliding_paths() + [f"D:/Trip/extra_{i}.jpg" for i in range(300)]
    at = AppTest.from_function(_gallery_app, args=(paths,), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len([b for b in at.button if b.label == "Detail"]) == len(paths)
