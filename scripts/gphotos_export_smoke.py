"""Standalone smoke test: export a folder of images to a Google Photos album.

Independent of Albumify. Uploads every image in --dir into a new (or reused)
app-owned album, then prints the result. First run opens a browser to consent to
the Library 'appendonly' scope (separate token from the picker).

Usage:
    .venv/Scripts/python scripts/gphotos_export_smoke.py --dir ./_gphotos_smoke/cache --title "sim-bench test"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gphotos.export_album import export_album_to_google_photos

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Export images to a Google Photos album")
    ap.add_argument("--client-secret", default="client_secret.json")
    ap.add_argument("--dir", required=True, help="folder of images to export")
    ap.add_argument("--title", default="sim-bench export")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    images = sorted(
        p for p in Path(args.dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"no images found in {args.dir}")
        return 1
    print(f"[1/2] exporting {len(images)} image(s) -> album '{args.title}'...")

    res = export_album_to_google_photos(
        images,
        args.title,
        client_secret=args.client_secret,
        manifest_path=args.manifest,
        on_progress=lambda i, n: print(f"      uploaded {i}/{n}", end="\r"),
    )

    print()
    print(f"[2/2] album_id={res.album_id}")
    print(f"      uploaded={res.uploaded} created={res.created} "
          f"skipped={res.skipped} failed={len(res.failed)}")
    if res.failed:
        for f in res.failed:
            print(f"        FAILED {f}")
    return 0 if res.created or res.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
