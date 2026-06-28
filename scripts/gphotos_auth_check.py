"""Standalone check: authenticate to Google Photos and print token status.

No Albumify, no API, no DB. Opens a browser for consent on first run, then
caches the token so later runs are silent.

Usage:
    .venv/Scripts/python scripts/gphotos_auth_check.py --client-secret client_secret.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow running as a plain script (not just via -m)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gphotos.auth import DEFAULT_TOKEN_PATH, PICKER_SCOPE, GooglePhotosAuth


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Check Google Photos OAuth")
    ap.add_argument("--client-secret", default="client_secret.json")
    ap.add_argument("--token", default=str(DEFAULT_TOKEN_PATH))
    args = ap.parse_args()

    auth = GooglePhotosAuth(
        args.client_secret, scopes=[PICKER_SCOPE], token_path=args.token
    )
    creds = auth.get_credentials()
    print("token OK")
    print(f"  valid:   {creds.valid}")
    print(f"  scopes:  {creds.scopes}")
    print(f"  expires: {creds.expiry}")
    print(f"  stored:  {args.token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
