"""Main debugger class."""

import json
import numpy as np
from pathlib import Path
from typing import List
from .models import TestResult
from .hypothesis_tests import HypothesisTests
from .reporting import Reporter


class FaceCropDebugger:
    """Systematic debugger for face crop embedding offset issue."""

    def __init__(
        self,
        crops_dir: Path,
        stored_embeddings_path: Path,
        fresh_embeddings_path: Path,
        stored_metadata_path: Path,
        fresh_metadata_path: Path
    ):
        self.crops_dir = Path(crops_dir)
        self.stored_embeddings_path = Path(stored_embeddings_path)
        self.fresh_embeddings_path = Path(fresh_embeddings_path)
        self.stored_metadata_path = Path(stored_metadata_path)
        self.fresh_metadata_path = Path(fresh_metadata_path)

        # Loaded data
        self.stored_embeddings = None
        self.fresh_embeddings = None
        self.stored_metadata = None
        self.fresh_metadata = None
        self.crop_files = None

        # Initialize test suite
        self.tests = HypothesisTests(self)

        self._load_data()

    def _load_data(self):
        """Load all necessary data files."""
        print("Loading data files...")

        self.stored_embeddings = np.load(self.stored_embeddings_path)
        self.fresh_embeddings = np.load(self.fresh_embeddings_path)
        print(f"  Stored embeddings: {self.stored_embeddings.shape}")
        print(f"  Fresh embeddings: {self.fresh_embeddings.shape}")

        with open(self.stored_metadata_path, 'r') as f:
            self.stored_metadata = json.load(f)
        with open(self.fresh_metadata_path, 'r') as f:
            self.fresh_metadata = json.load(f)
        print(f"  Stored metadata: {len(self.stored_metadata)} entries")
        print(f"  Fresh metadata: {len(self.fresh_metadata)} entries")

        self.crop_files = sorted(self.crops_dir.glob("face_*_aligned.jpg"))
        print(f"  Crop files: {len(self.crop_files)} files")
        print()

    def run_all_tests(self) -> List[TestResult]:
        """Run all hypothesis tests in order."""
        results = []

        # Run tests in logical order
        results.append(self.tests.test_h1_gap_in_crop_files())
        results.append(self.tests.test_h2_string_vs_numeric_sort())
        results.append(self.tests.test_h3_metadata_index_mismatch())
        results.append(self.tests.test_h4_embedding_extraction_order())
        results.append(self.tests.test_h5_staleness_check())
        results.append(self.tests.test_h6_offset_pattern_verification())

        return results

    def print_report(self, results: List[TestResult]):
        """Print formatted report."""
        Reporter.print_report(results)
