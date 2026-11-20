"""Which file a thumbnail is made from (SIGHTING-152 C5).

The API sends each photo's 768 px copy as ``display_path`` next to its original ``path``. The client records the pair
when it parses an image, and every thumbnail loader asks :func:`source` for the file to open, so pages that only
hold an original path (step decisions, comparisons, people) still read the small copy. Unknown paths come back
unchanged: older runs have no copies, and the original is still right for them.
"""
from typing import Dict, Optional

_MAX_PATHS = 50_000            # a few large albums; the oldest pairs go first
_copies: Dict[str, str] = {}


def _key(path: str) -> str:
    return str(path).replace("\\", "/")


def remember(path: str, display_path: Optional[str]) -> None:
    if not path or not display_path:
        return
    key = _key(path)
    _copies.pop(key, None)
    _copies[key] = display_path
    while len(_copies) > _MAX_PATHS:
        _copies.pop(next(iter(_copies)))


def source(path: str) -> str:
    """The 768 px copy of ``path`` if one is known, else ``path``."""
    return _copies.get(_key(path), path) if path else path
