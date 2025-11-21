"""
Extract reason texts from PhotoTriage review JSONs.

This script:
1. Loads all review JSON files
2. Extracts reason texts and comparison metadata
3. Saves raw extracted data for analysis
4. Generates summary statistics

Usage:
    python scripts/phototriage/01_extract_reasons.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass, asdict
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ReviewComparison:
    """A single pairwise comparison from PhotoTriage reviews."""
    series_id: str
    compare_id_1: int  # 0-based index
    compare_id_2: int  # 0-based index
    compare_file_1: str
    compare_file_2: str
    user_choice: str  # "LEFT" or "RIGHT"
    reason_text: str
    review_file: str
    review_index: int  # Index within the review file


class ReviewExtractor:
    """Extract comparisons and reasons from PhotoTriage review JSONs."""

    def __init__(self, reviews_dir: Path):
        """
        Initialize extractor.

        Args:
            reviews_dir: Directory containing review JSON files
        """
        self.reviews_dir = Path(reviews_dir)

        if not self.reviews_dir.exists():
            raise FileNotFoundError(f"Reviews directory not found: {reviews_dir}")

        self.comparisons: List[ReviewComparison] = []
        self.errors: List[Dict] = []

    def extract_all(self) -> Tuple[List[ReviewComparison], Dict]:
        """
        Extract all comparisons from review JSONs.

        Returns:
            (comparisons, statistics)
        """
        logger.info(f"Scanning for JSON files in {self.reviews_dir}")

        json_files = sorted(self.reviews_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        for i, json_file in enumerate(json_files, 1):
            if i % 500 == 0:
                logger.info(f"Processing file {i}/{len(json_files)}")

            try:
                self._extract_from_file(json_file)
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
                self.errors.append({
                    'file': json_file.name,
                    'error': str(e)
                })

        # Generate statistics
        stats = self._compute_statistics()

        logger.info(f"Extraction complete:")
        logger.info(f"  Total comparisons: {len(self.comparisons)}")
        logger.info(f"  Unique series: {stats['unique_series']}")
        logger.info(f"  Files with errors: {len(self.errors)}")

        return self.comparisons, stats

    def _extract_from_file(self, json_file: Path) -> None:
        """Extract comparisons from a single JSON file."""
        # Series ID is the filename without extension
        series_id = json_file.stem

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        reviews = data.get('reviews', [])

        for review_idx, review in enumerate(reviews):
            # Extract comparison metadata
            compare_id_1 = review.get('compareID1')
            compare_id_2 = review.get('compareID2')
            compare_file_1 = review.get('compareFile1', '')
            compare_file_2 = review.get('compareFile2', '')
            user_choice = review.get('userChoice', '')

            # Extract reason text
            # Reason is always array: [empty_string, actual_reason]
            reason = review.get('reason', ['', ''])

            if len(reason) < 2:
                logger.warning(
                    f"{json_file.name} review {review_idx}: "
                    f"Reason array has < 2 elements"
                )
                reason_text = ''
            else:
                reason_text = reason[1].strip()

            # Skip if no reason text
            if not reason_text:
                continue

            # Create comparison object
            comparison = ReviewComparison(
                series_id=series_id,
                compare_id_1=compare_id_1,
                compare_id_2=compare_id_2,
                compare_file_1=compare_file_1,
                compare_file_2=compare_file_2,
                user_choice=user_choice,
                reason_text=reason_text,
                review_file=json_file.name,
                review_index=review_idx
            )

            self.comparisons.append(comparison)

    def _compute_statistics(self) -> Dict:
        """Compute statistics about extracted comparisons."""
        if not self.comparisons:
            return {
                'unique_series': 0,
                'avg_comparisons_per_series': 0,
                'avg_reason_length': 0,
                'user_choice_distribution': {}
            }

        # Count unique series
        series_ids = set(c.series_id for c in self.comparisons)

        # Comparisons per series
        series_counts = Counter(c.series_id for c in self.comparisons)
        avg_comparisons = sum(series_counts.values()) / len(series_counts)

        # Average reason length
        reason_lengths = [len(c.reason_text) for c in self.comparisons]
        avg_reason_length = sum(reason_lengths) / len(reason_lengths)

        # User choice distribution
        user_choices = Counter(c.user_choice for c in self.comparisons)

        return {
            'unique_series': len(series_ids),
            'total_comparisons': len(self.comparisons),
            'avg_comparisons_per_series': avg_comparisons,
            'max_comparisons_in_series': max(series_counts.values()),
            'min_comparisons_in_series': min(series_counts.values()),
            'avg_reason_length': avg_reason_length,
            'min_reason_length': min(reason_lengths),
            'max_reason_length': max(reason_lengths),
            'user_choice_distribution': dict(user_choices)
        }

    def save_comparisons(self, output_file: Path) -> None:
        """
        Save comparisons to JSONL file.

        Args:
            output_file: Output JSONL path
        """
        logger.info(f"Saving {len(self.comparisons)} comparisons to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for comparison in self.comparisons:
                json.dump(asdict(comparison), f)
                f.write('\n')

        logger.info(f"Saved to {output_file}")

    def save_statistics(self, output_file: Path, stats: Dict) -> None:
        """Save statistics to JSON file."""
        logger.info(f"Saving statistics to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved to {output_file}")

    def save_as_dataframe(self, output_file: Path) -> None:
        """Save comparisons as CSV for easy inspection."""
        logger.info(f"Saving as DataFrame to {output_file}")

        df = pd.DataFrame([asdict(c) for c in self.comparisons])

        # Add computed columns
        df['reason_length'] = df['reason_text'].str.len()
        df['reason_word_count'] = df['reason_text'].str.split().str.len()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        logger.info(f"Saved {len(df)} rows to {output_file}")


def main():
    """Main extraction pipeline."""
    # Paths
    reviews_dir = Path("D:/Similar Images/automatic_triage_photo_series/train_val/reviews_trainval/reviews_trainval")
    output_dir = Path("data/phototriage")

    # Check if reviews directory exists
    if not reviews_dir.exists():
        logger.error(f"Reviews directory not found: {reviews_dir}")
        logger.error("Please update the path in this script to match your system")
        return

    # Create extractor
    logger.info("="*60)
    logger.info("PhotoTriage Reason Extraction")
    logger.info("="*60)

    extractor = ReviewExtractor(reviews_dir)

    # Extract all comparisons
    comparisons, stats = extractor.extract_all()

    if not comparisons:
        logger.error("No comparisons extracted! Check your data path.")
        return

    # Save outputs
    extractor.save_comparisons(output_dir / "raw_comparisons.jsonl")
    extractor.save_statistics(output_dir / "extraction_stats.json", stats)
    extractor.save_as_dataframe(output_dir / "raw_comparisons.csv")

    # Print summary
    logger.info("="*60)
    logger.info("Extraction Summary")
    logger.info("="*60)
    logger.info(f"Total comparisons extracted: {stats['total_comparisons']}")
    logger.info(f"Unique series: {stats['unique_series']}")
    logger.info(f"Avg comparisons/series: {stats['avg_comparisons_per_series']:.1f}")
    logger.info(f"Avg reason length: {stats['avg_reason_length']:.1f} chars")
    logger.info(f"User choice distribution: {stats['user_choice_distribution']}")
    logger.info("="*60)

    # Show some examples
    logger.info("\nExample reasons:")
    for i, comp in enumerate(comparisons[:10], 1):
        logger.info(f"{i}. [{comp.user_choice}] {comp.reason_text[:80]}...")


if __name__ == "__main__":
    main()
