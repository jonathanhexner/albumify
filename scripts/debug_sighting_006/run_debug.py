#!/usr/bin/env python3
"""
Main entry point for SIGHTING-006 debug script.

Usage:
    python scripts/debug_sighting_006/run_debug.py \\
        --crops results/Google_Germany/face_crops \\
        --stored-embeddings "results/Google_Germany/embeddings_2026-*.npy" \\
        --fresh-embeddings "results/Google_Germany/embeddings_FRESH_*.npy" \\
        --stored-metadata "results/Google_Germany/embeddings_metadata_2026-*.json" \\
        --fresh-metadata "results/Google_Germany/embeddings_metadata_FRESH_*.json"
"""

import argparse
from pathlib import Path
from .debugger import FaceCropDebugger
from .helpers import resolve_glob_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Debug face crop embedding offset issue (SIGHTING-006)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--crops", required=True, help="Path to face_crops directory")
    parser.add_argument("--stored-embeddings", required=True, help="Stored embeddings NPY (glob OK)")
    parser.add_argument("--fresh-embeddings", required=True, help="Fresh embeddings NPY (glob OK)")
    parser.add_argument("--stored-metadata", required=True, help="Stored metadata JSON (glob OK)")
    parser.add_argument("--fresh-metadata", required=True, help="Fresh metadata JSON (glob OK)")

    args = parser.parse_args()

    # Resolve glob patterns
    stored_embeddings_path = resolve_glob_path(args.stored_embeddings)
    fresh_embeddings_path = resolve_glob_path(args.fresh_embeddings)
    stored_metadata_path = resolve_glob_path(args.stored_metadata)
    fresh_metadata_path = resolve_glob_path(args.fresh_metadata)

    print(f"Using files:")
    print(f"  Crops dir: {args.crops}")
    print(f"  Stored embeddings: {stored_embeddings_path}")
    print(f"  Fresh embeddings: {fresh_embeddings_path}")
    print(f"  Stored metadata: {stored_metadata_path}")
    print(f"  Fresh metadata: {fresh_metadata_path}")
    print()

    # Run debug
    debugger = FaceCropDebugger(
        crops_dir=Path(args.crops),
        stored_embeddings_path=stored_embeddings_path,
        fresh_embeddings_path=fresh_embeddings_path,
        stored_metadata_path=stored_metadata_path,
        fresh_metadata_path=fresh_metadata_path,
    )

    results = debugger.run_all_tests()
    debugger.print_report(results)


if __name__ == "__main__":
    main()
