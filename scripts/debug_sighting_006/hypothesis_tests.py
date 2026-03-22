"""Hypothesis test implementations."""

import random
from pathlib import Path
from datetime import datetime
from typing import List
from .models import TestResult, Verdict
from .helpers import extract_face_id_from_filename, compute_embedding_distance


class HypothesisTests:
    """Container for all hypothesis test methods."""

    def __init__(self, debugger):
        """Initialize with reference to debugger instance."""
        self.debugger = debugger

    def test_h1_gap_in_crop_files(self) -> TestResult:
        """
        H1: First 2 crop files don't exist (face_0000, face_0001 missing).
        """
        face_ids = [extract_face_id_from_filename(f.name) for f in self.debugger.crop_files]
        face_ids_sorted = sorted(face_ids)

        has_face_0000 = 0 in face_ids
        has_face_0001 = 1 in face_ids
        actual_first = face_ids_sorted[0] if face_ids_sorted else None

        expected_range = set(range(min(face_ids), max(face_ids) + 1))
        actual_set = set(face_ids)
        missing = sorted(expected_range - actual_set)

        evidence = {
            "Total crop files": len(self.debugger.crop_files),
            "Expected first face_id": 0,
            "Actual first face_id": actual_first,
            "face_0000 exists": has_face_0000,
            "face_0001 exists": has_face_0001,
            "Missing face_ids": missing[:10] if len(missing) <= 10 else f"{missing[:10]} ... ({len(missing)} total)",
            "First 5 face_ids": face_ids_sorted[:5],
            "Last 5 face_ids": face_ids_sorted[-5:],
        }

        if not has_face_0000 and not has_face_0001 and actual_first == 2:
            verdict = Verdict.FAIL
            conclusion = "face_0000 and face_0001 are MISSING. This explains the +2 offset perfectly."
            recommendation = "Investigate why first 2 faces were not saved. Check filtering logic in crop generation."
        elif missing:
            verdict = Verdict.SUSPICIOUS
            conclusion = f"Found {len(missing)} missing face_ids in sequence. May indicate filtering or indexing issues."
            recommendation = "Review face filtering logic to understand why some faces were skipped."
        else:
            verdict = Verdict.PASS
            conclusion = "All expected crop files exist in sequence starting from face_0000."
            recommendation = None

        return TestResult(
            hypothesis="H1: First 2 crop files don't exist (face_0000, face_0001 missing)",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )

    def test_h2_string_vs_numeric_sort(self) -> TestResult:
        """
        H2: String sorting vs numeric sorting causes order mismatch.
        """
        string_sorted = sorted([f.name for f in self.debugger.crop_files])
        numeric_sorted = sorted(
            [f.name for f in self.debugger.crop_files],
            key=lambda x: extract_face_id_from_filename(x)
        )

        differences = []
        for i in range(min(20, len(string_sorted))):
            if string_sorted[i] != numeric_sorted[i]:
                differences.append({
                    "position": i,
                    "string_sort": string_sorted[i],
                    "numeric_sort": numeric_sorted[i]
                })

        evidence = {
            "String sort first 5": [extract_face_id_from_filename(f) for f in string_sorted[:5]],
            "Numeric sort first 5": [extract_face_id_from_filename(f) for f in numeric_sorted[:5]],
            "Differences in first 20": len(differences),
            "Example differences": differences[:3] if differences else "None"
        }

        if differences:
            verdict = Verdict.FAIL
            conclusion = f"String vs numeric sorting produces different order ({len(differences)} differences)."
            recommendation = "Check if code uses string sort when it should use numeric sort."
        else:
            verdict = Verdict.PASS
            conclusion = "String and numeric sort produce identical order. No sorting issues."
            recommendation = None

        return TestResult(
            hypothesis="H2: String sorting vs numeric sorting causes order mismatch",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )

    def test_h3_metadata_index_mismatch(self) -> TestResult:
        """
        H3: Metadata array indices don't match crop filenames.
        """
        fresh_face_ids = [entry.get('face_id') for entry in self.debugger.fresh_metadata]
        crop_face_ids = sorted([extract_face_id_from_filename(f.name) for f in self.debugger.crop_files])

        comparisons = []
        for i in range(min(10, len(fresh_face_ids), len(crop_face_ids))):
            meta_face_id = fresh_face_ids[i]
            expected_face_id = crop_face_ids[i]
            comparisons.append({
                "index": i,
                "metadata_face_id": meta_face_id,
                "crop_file_face_id": expected_face_id,
                "match": meta_face_id == expected_face_id
            })

        offsets = []
        for comp in comparisons:
            if comp["metadata_face_id"] is not None and comp["crop_file_face_id"] is not None:
                offset = comp["metadata_face_id"] - comp["crop_file_face_id"]
                offsets.append(offset)

        consistent_offset = offsets[0] if offsets and len(set(offsets)) == 1 else None

        evidence = {
            "Metadata entries": len(fresh_face_ids),
            "Crop files": len(crop_face_ids),
            "Comparison table": comparisons,
            "Consistent offset": consistent_offset if consistent_offset is not None else "None (varying)"
        }

        if consistent_offset == 0:
            verdict = Verdict.PASS
            conclusion = "Metadata face_ids perfectly match crop file face_ids."
            recommendation = None
        elif consistent_offset is not None:
            verdict = Verdict.FAIL
            conclusion = f"Metadata has consistent offset of {consistent_offset:+d}."
            recommendation = f"Fix metadata generation - currently off by {consistent_offset}."
        else:
            verdict = Verdict.SUSPICIOUS
            conclusion = "Metadata face_ids have VARYING offsets. Inconsistent indexing."
            recommendation = "Investigate metadata generation logic for non-deterministic indexing."

        return TestResult(
            hypothesis="H3: Metadata array indices don't match crop filenames",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )

    def test_h4_embedding_extraction_order(self) -> TestResult:
        """
        H4: Embeddings extracted in different order than crop files.
        """
        fresh_face_ids = [entry.get('face_id') for entry in self.debugger.fresh_metadata]
        crop_face_ids = sorted([extract_face_id_from_filename(f.name) for f in self.debugger.crop_files])

        order_matches = fresh_face_ids == crop_face_ids

        first_mismatch_idx = None
        for i in range(min(len(fresh_face_ids), len(crop_face_ids))):
            if fresh_face_ids[i] != crop_face_ids[i]:
                first_mismatch_idx = i
                break

        evidence = {
            "Fresh metadata face_ids (first 10)": fresh_face_ids[:10],
            "Sorted crop face_ids (first 10)": crop_face_ids[:10],
            "Orders match": order_matches,
            "First mismatch at index": first_mismatch_idx if first_mismatch_idx is not None else "No mismatch"
        }

        if order_matches:
            verdict = Verdict.PASS
            conclusion = "Embedding extraction order matches sorted crop file order."
            recommendation = None
        else:
            verdict = Verdict.FAIL
            conclusion = f"Extraction order differs (first mismatch at index {first_mismatch_idx})."
            recommendation = "Check if regenerate_embeddings_from_crops.py sorts files correctly."

        return TestResult(
            hypothesis="H4: Embeddings extracted in different order than crop files",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )

    def test_h5_staleness_check(self) -> TestResult:
        """
        H5: Stored embeddings are from different/older crop set.
        """
        crops_mtime = max([f.stat().st_mtime for f in self.debugger.crop_files])
        stored_mtime = self.debugger.stored_embeddings_path.stat().st_mtime
        fresh_mtime = self.debugger.fresh_embeddings_path.stat().st_mtime

        crops_time = datetime.fromtimestamp(crops_mtime).isoformat()
        stored_time = datetime.fromtimestamp(stored_mtime).isoformat()
        fresh_time = datetime.fromtimestamp(fresh_mtime).isoformat()

        crops_count = len(self.debugger.crop_files)
        stored_count = len(self.debugger.stored_embeddings)
        fresh_count = len(self.debugger.fresh_embeddings)

        evidence = {
            "Crop files count": crops_count,
            "Stored embeddings count": stored_count,
            "Fresh embeddings count": fresh_count,
            "Newest crop timestamp": crops_time,
            "Stored embeddings timestamp": stored_time,
            "Fresh embeddings timestamp": fresh_time,
        }

        if crops_count != stored_count:
            verdict = Verdict.SUSPICIOUS
            conclusion = f"Count mismatch: {crops_count} crops but {stored_count} stored embeddings."
            recommendation = "Stored embeddings may be from different crop set."
        elif crops_mtime > stored_mtime:
            verdict = Verdict.INFO
            conclusion = "Crops newer than stored embeddings. May be stale but counts match."
            recommendation = None
        else:
            verdict = Verdict.PASS
            conclusion = "Counts match and timestamps reasonable."
            recommendation = None

        return TestResult(
            hypothesis="H5: Stored embeddings are from different/older crop set",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )

    def test_h6_offset_pattern_verification(self) -> TestResult:
        """
        H6: Verify +2 offset pattern across sample of faces.
        """
        max_idx = min(len(self.debugger.stored_embeddings),
                     len(self.debugger.fresh_embeddings) - 2) - 1
        sample_size = min(20, max_idx)

        random.seed(42)
        sample_indices = sorted(random.sample(range(max_idx), sample_size)) if max_idx >= sample_size else range(max_idx)

        results = []
        for idx in sample_indices:
            dist_n = compute_embedding_distance(
                self.debugger.stored_embeddings[idx],
                self.debugger.fresh_embeddings[idx]
            )
            dist_n2 = compute_embedding_distance(
                self.debugger.stored_embeddings[idx],
                self.debugger.fresh_embeddings[idx + 2]
            )

            results.append({
                "idx": idx,
                "dist_+0": f"{dist_n:.4f}",
                "dist_+2": f"{dist_n2:.4f}",
                "offset_2_is_better": dist_n2 < dist_n
            })

        offset_2_better_count = sum(1 for r in results if r["offset_2_is_better"])

        evidence = {
            "Sample size": len(results),
            "Times offset +2 is better": f"{offset_2_better_count}/{len(results)}",
            "Percentage": f"{100 * offset_2_better_count / len(results):.1f}%",
            "Sample results (first 5)": results[:5]
        }

        if offset_2_better_count == len(results):
            verdict = Verdict.FAIL
            conclusion = "100% of samples show +2 offset. Pattern is UNIVERSAL."
            recommendation = "Root cause affects ALL faces systematically."
        elif offset_2_better_count > 0.8 * len(results):
            verdict = Verdict.SUSPICIOUS
            conclusion = f"{100 * offset_2_better_count / len(results):.1f}% show +2 offset."
            recommendation = "Offset is dominant but not universal."
        else:
            verdict = Verdict.PASS
            conclusion = f"No consistent +2 pattern ({100 * offset_2_better_count / len(results):.1f}%)."
            recommendation = None

        return TestResult(
            hypothesis="H6: Verify +2 offset pattern is consistent across faces",
            verdict=verdict,
            evidence=evidence,
            conclusion=conclusion,
            recommendation=recommendation
        )
