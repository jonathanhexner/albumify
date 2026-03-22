"""Report generation and formatting."""

from typing import List
from .models import TestResult, Verdict


class Reporter:
    """Handles report generation and output formatting."""

    @staticmethod
    def print_report(results: List[TestResult]):
        """Print formatted report of all test results."""
        print("\n" + "=" * 80)
        print("SIGHTING-006 DEBUG REPORT: Face Crop Embedding Offset (+2)")
        print("=" * 80)
        print()

        for i, result in enumerate(results, 1):
            Reporter._print_test_result(i, result)

        Reporter._print_summary(results)
        Reporter._print_recommendations(results)

    @staticmethod
    def _print_test_result(test_num: int, result: TestResult):
        """Print individual test result."""
        print(f"\n{'='*80}")
        print(f"Test {test_num}: {result.hypothesis}")
        print(f"{'='*80}")
        print(f"Verdict: {result.verdict.value}")
        print(f"\nConclusion: {result.conclusion}")

        print(f"\nEvidence:")
        for key, value in result.evidence.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: {value[:5]} ... (showing first 5)")
            else:
                print(f"  {key}: {value}")

        if result.recommendation:
            print(f"\n💡 Recommendation: {result.recommendation}")

    @staticmethod
    def _print_summary(results: List[TestResult]):
        """Print summary section."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        failed = [r for r in results if r.verdict == Verdict.FAIL]
        suspicious = [r for r in results if r.verdict == Verdict.SUSPICIOUS]
        passed = [r for r in results if r.verdict == Verdict.PASS]

        if failed:
            print(f"\n❌ FAILED TESTS ({len(failed)}):")
            for r in failed:
                print(f"  - {r.hypothesis}")

        if suspicious:
            print(f"\n⚠️  SUSPICIOUS TESTS ({len(suspicious)}):")
            for r in suspicious:
                print(f"  - {r.hypothesis}")

        if passed:
            print(f"\n✅ PASSED TESTS ({len(passed)}):")
            for r in passed:
                print(f"  - {r.hypothesis}")

    @staticmethod
    def _print_recommendations(results: List[TestResult]):
        """Print actionable recommendations based on results."""
        print("\n" + "=" * 80)
        print("RECOMMENDED NEXT STEPS")
        print("=" * 80)
        print()

        # Find key test results
        h1_result = next((r for r in results if "H1:" in r.hypothesis), None)
        h3_result = next((r for r in results if "H3:" in r.hypothesis), None)
        h6_result = next((r for r in results if "H6:" in r.hypothesis), None)

        # Scenario 1: Missing crop files
        if h1_result and h1_result.verdict == Verdict.FAIL:
            print("ROOT CAUSE IDENTIFIED:")
            print("  First 2 faces (face_0000, face_0001) are MISSING from crops directory.")
            print("  This creates systematic +2 offset: stored[N] → fresh[N+2]")
            print()
            print("WHY THIS HAPPENED:")
            print("  - Faces 0 and 1 were filtered out (quality/size/frontal score)")
            print("  - Crop saving used loop counter for filenames instead of face_id")
            print("  - Metadata array still indexed from 0, creating mismatch")
            print()
            print("IMMEDIATE FIX:")
            print("  1. Re-run crop generation with face_id → filename mapping (RECOMMENDED)")
            print("  2. OR: Regenerate embeddings with adjusted metadata")
            print()
            print("PREVENTION:")
            print("  1. Use face_id from metadata for filenames, NOT loop counter")
            print("  2. Add assertion: len(saved_crops) == expected_count")
            print("  3. Add test: test_crop_filename_matches_face_id()")

        # Scenario 2: Metadata indexing issue
        elif h3_result and h3_result.verdict == Verdict.FAIL:
            offset = h3_result.evidence.get('Consistent offset', 'unknown')
            print("ROOT CAUSE IDENTIFIED:")
            print("  Metadata array indexing doesn't match crop filenames.")
            print(f"  Detected consistent offset: {offset}")
            print()
            print("IMMEDIATE FIX:")
            print("  1. Fix metadata generation to use crop file face_ids directly")
            print("  2. Regenerate metadata JSON with corrected indices")

        # Scenario 3: Pattern confirmed but unclear root cause
        elif h6_result and h6_result.verdict == Verdict.FAIL:
            print("SYMPTOM CONFIRMED:")
            print("  +2 offset pattern exists across all sampled faces.")
            print("  Root cause not yet identified from automated tests.")
            print()
            print("NEXT STEPS:")
            print("  1. Manually inspect face_0569.jpg and face_0571.jpg")
            print("  2. Check if they contain expected images from metadata")
            print("  3. Review crop generation code for indexing logic")
            print("  4. Check filtering logs to see which faces were skipped")

        else:
            print("NO CLEAR ROOT CAUSE IDENTIFIED:")
            print("  Tests did not identify a definitive cause of the offset.")
            print()
            print("RECOMMENDATIONS:")
            print("  1. Review suspicious tests for clues")
            print("  2. Manually trace through crop generation code")
            print("  3. Add debug prints to crop saving and metadata generation")
            print("  4. Consider regenerating all data from scratch with logging")
