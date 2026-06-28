"""Standalone smoke test: import photos via the Google Photos Picker.

Opens a browser, lets you pick photos, downloads them to a local cache dir, and
prints the result. Independent of Albumify -- no FastAPI, no Streamlit, no
sim_bench.db. This is the P3 acceptance gate for spec-092.

Usage:
    .venv/Scripts/python scripts/gphotos_picker_smoke.py \
        --client-secret client_secret.json --out ./_gphotos_smoke/cache
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gphotos.auth import GooglePhotosAuth
from gphotos.cache import DownloadCache
from gphotos.picker import PickerClient
from gphotos.types import PickerSession


def _announce(sess: PickerSession) -> None:
    print(f"  waiting... pick photos in the browser, then come back ({sess.id})")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Google Photos Picker smoke test")
    ap.add_argument("--client-secret", default="client_secret.json")
    ap.add_argument("--out", default="./_gphotos_smoke/cache")
    args = ap.parse_args()

    creds = GooglePhotosAuth(args.client_secret).get_credentials()
    client = PickerClient(creds)

    sess = client.create_session()
    print(f"[1/4] session created: {sess.id}")
    print(f"[2/4] opening picker: {sess.picker_uri}")
    webbrowser.open(sess.picker_uri)

    ready = client.poll_until_ready(sess, on_wait=_announce)
    items = client.list_media_items(ready.id)
    print(f"[3/4] picked {len(items)} item(s)")

    downloaded = DownloadCache(args.out).materialize(client, items)
    print(f"[4/4] downloaded {len(downloaded)} file(s) -> {args.out}")
    for d in downloaded:
        print(f"    {Path(d.local_path).name}")

    client.delete_session(ready.id)
    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())
