"""
Map reason texts to attribute labels using the AttributeMapper.

This script:
1. Loads extracted comparisons
2. Maps each reason to attribute labels
3. Saves enriched comparisons with attributes
4. Generates attribute statistics

Usage:
    python scripts/phototriage/03_map_attributes.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict
from dataclasses import asdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sim_bench.phototriage.attribute_mapper import AttributeMapper, AttributeLabel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttributeLabelingPipeline:
    """Label comparisons with attributes."""

    def __init__(self, comparisons_file: Path):
        """
        Initialize pipeline.

        Args:
            comparisons_file: Path to raw_comparisons.jsonl
        """
        self.comparisons_file = Path(comparisons_file)
        self.comparisons: List[Dict] = []
        self.labeled_comparisons: List[Dict] = []

        # Create mapper
        self.mapper = AttributeMapper()

    def load_comparisons(self) -> None:
        """Load comparisons from JSONL file."""
        logger.info(f"Loading comparisons from {self.comparisons_file}")

        with open(self.comparisons_file, 'r', encoding='utf-8') as f:
            self.comparisons = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.comparisons)} comparisons")

    def label_all_comparisons(self) -> None:
        """Label all comparisons with attributes."""
        logger.info("Labeling comparisons with attributes...")

        for i, comparison in enumerate(self.comparisons, 1):
            if i % 1000 == 0:
                logger.info(f"Processing comparison {i}/{len(self.comparisons)}")

            # Extract attributes
            attributes = self.mapper.map_reason_to_attributes(
                reason_text=comparison['reason_text'],
                user_choice=comparison['user_choice']
            )

            # Convert to dictionaries
            attribute_dicts = [
                {
                    'name': attr.name,
                    'winner': attr.winner,
                    'confidence': attr.confidence,
                    'reason_snippet': attr.reason_snippet,
                    'category': attr.category.value
                }
                for attr in attributes
            ]

            # Create enriched comparison
            labeled = {
                **comparison,  # Include all original fields
                'attributes': attribute_dicts,
                'num_attributes': len(attribute_dicts)
            }

            self.labeled_comparisons.append(labeled)

        logger.info(f"Labeled {len(self.labeled_comparisons)} comparisons")

    def compute_statistics(self) -> Dict:
        """Compute statistics about attribute labels."""
        logger.info("Computing attribute statistics...")

        # Count attributes
        attribute_counts = Counter()
        category_counts = Counter()
        confidence_sums = defaultdict(float)
        confidence_counts = defaultdict(int)

        # Track multi-attribute pairs
        num_multi_attribute = sum(1 for c in self.labeled_comparisons if c['num_attributes'] > 1)
        num_single_attribute = sum(1 for c in self.labeled_comparisons if c['num_attributes'] == 1)
        num_no_attributes = sum(1 for c in self.labeled_comparisons if c['num_attributes'] == 0)

        # Collect examples
        attribute_examples = defaultdict(list)

        for comparison in self.labeled_comparisons:
            for attr in comparison['attributes']:
                attr_name = attr['name']
                category = attr['category']

                # Count
                attribute_counts[attr_name] += 1
                category_counts[category] += 1

                # Track confidence
                confidence_sums[attr_name] += attr['confidence']
                confidence_counts[attr_name] += 1

                # Collect examples (first 5)
                if len(attribute_examples[attr_name]) < 5:
                    attribute_examples[attr_name].append({
                        'reason': comparison['reason_text'],
                        'snippet': attr['reason_snippet'],
                        'winner': attr['winner'],
                        'confidence': attr['confidence']
                    })

        # Compute averages
        total_labels = sum(attribute_counts.values())

        attribute_stats = {}
        for attr_name, count in attribute_counts.items():
            avg_confidence = confidence_sums[attr_name] / confidence_counts[attr_name]

            attribute_stats[attr_name] = {
                'count': count,
                'percentage': 100.0 * count / len(self.labeled_comparisons),
                'avg_confidence': avg_confidence,
                'examples': attribute_examples[attr_name]
            }

        # Overall stats
        stats = {
            'total_comparisons': len(self.labeled_comparisons),
            'total_attribute_labels': total_labels,
            'avg_attributes_per_comparison': total_labels / len(self.labeled_comparisons),
            'comparisons_with_no_attributes': num_no_attributes,
            'comparisons_with_one_attribute': num_single_attribute,
            'comparisons_with_multiple_attributes': num_multi_attribute,
            'attribute_statistics': attribute_stats,
            'category_distribution': dict(category_counts)
        }

        return stats

    def save_labeled_comparisons(self, output_file: Path) -> None:
        """Save labeled comparisons to JSONL."""
        logger.info(f"Saving labeled comparisons to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for comparison in self.labeled_comparisons:
                json.dump(comparison, f)
                f.write('\n')

        logger.info(f"Saved {len(self.labeled_comparisons)} labeled comparisons")

    def save_statistics(self, output_file: Path, stats: Dict) -> None:
        """Save statistics to JSON."""
        logger.info(f"Saving statistics to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {output_file}")

    def print_summary(self, stats: Dict) -> None:
        """Print summary of attribute labeling."""
        logger.info("="*70)
        logger.info("ATTRIBUTE LABELING SUMMARY")
        logger.info("="*70)

        logger.info(f"\nTotal comparisons: {stats['total_comparisons']}")
        logger.info(f"Total attribute labels: {stats['total_attribute_labels']}")
        logger.info(f"Avg labels/comparison: {stats['avg_attributes_per_comparison']:.2f}")

        logger.info(f"\nLabel distribution:")
        logger.info(f"  No attributes: {stats['comparisons_with_no_attributes']} "
                   f"({100.0 * stats['comparisons_with_no_attributes'] / stats['total_comparisons']:.1f}%)")
        logger.info(f"  One attribute: {stats['comparisons_with_one_attribute']} "
                   f"({100.0 * stats['comparisons_with_one_attribute'] / stats['total_comparisons']:.1f}%)")
        logger.info(f"  Multiple attributes: {stats['comparisons_with_multiple_attributes']} "
                   f"({100.0 * stats['comparisons_with_multiple_attributes'] / stats['total_comparisons']:.1f}%)")

        logger.info(f"\nCategory distribution:")
        for category, count in sorted(stats['category_distribution'].items(),
                                      key=lambda x: x[1], reverse=True):
            pct = 100.0 * count / stats['total_attribute_labels']
            logger.info(f"  {category:25s} {count:5d} ({pct:5.1f}%)")

        logger.info(f"\nAttribute statistics (sorted by count):")
        sorted_attrs = sorted(
            stats['attribute_statistics'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        logger.info(f"{'Attribute':<30} {'Count':>6} {'% Comp':>7} {'Avg Conf':>9}")
        logger.info("-"*70)

        for attr_name, attr_stats in sorted_attrs:
            logger.info(
                f"{attr_name:<30} "
                f"{attr_stats['count']:>6} "
                f"{attr_stats['percentage']:>6.1f}% "
                f"{attr_stats['avg_confidence']:>9.3f}"
            )

        # Show examples for top 3 attributes
        logger.info("\nExample mappings for top 3 attributes:")
        for attr_name, attr_stats in sorted_attrs[:3]:
            logger.info(f"\n{attr_name}:")
            for i, example in enumerate(attr_stats['examples'][:3], 1):
                logger.info(f"  {i}. \"{example['reason'][:70]}...\"")
                logger.info(f"     → Winner: {example['winner']}, "
                          f"Confidence: {example['confidence']:.2f}")
                logger.info(f"     Snippet: \"{example['snippet']}\"")

        logger.info("="*70)


def main():
    """Main attribute mapping pipeline."""
    # Paths
    comparisons_file = Path("data/phototriage/raw_comparisons.jsonl")
    labeled_output = Path("data/phototriage/labeled_comparisons.jsonl")
    stats_output = Path("data/phototriage/attribute_stats.json")

    # Check if input exists
    if not comparisons_file.exists():
        logger.error(f"Comparisons file not found: {comparisons_file}")
        logger.error("Please run 01_extract_reasons.py first")
        return

    # Create pipeline
    logger.info("="*70)
    logger.info("PhotoTriage Attribute Mapping")
    logger.info("="*70)

    pipeline = AttributeLabelingPipeline(comparisons_file)

    # Load comparisons
    pipeline.load_comparisons()

    # Label all comparisons
    pipeline.label_all_comparisons()

    # Compute statistics
    stats = pipeline.compute_statistics()

    # Save outputs
    pipeline.save_labeled_comparisons(labeled_output)
    pipeline.save_statistics(stats_output, stats)

    # Print summary
    pipeline.print_summary(stats)


if __name__ == "__main__":
    main()
