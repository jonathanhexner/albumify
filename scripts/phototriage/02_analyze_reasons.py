"""
Analyze extracted reason texts to inform attribute schema.

This script:
1. Loads extracted comparisons
2. Analyzes reason text patterns
3. Identifies common keywords and phrases
4. Clusters similar reasons
5. Generates visualizations and reports

Usage:
    python scripts/phototriage/02_analyze_reasons.py
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReasonAnalyzer:
    """Analyze PhotoTriage reason texts."""

    def __init__(self, comparisons_file: Path):
        """
        Initialize analyzer.

        Args:
            comparisons_file: Path to raw_comparisons.jsonl
        """
        self.comparisons_file = Path(comparisons_file)
        self.comparisons: List[Dict] = []
        self.reason_texts: List[str] = []

        # Stop words to filter out
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no',
            'yes', 'so', 'than', 'too', 'very', 'just', 'also', 'more', 'less',
            'one', 'two', 'both', 'all', 'some', 'any', 'each', 'every', 'other'
        }

    def load_comparisons(self) -> None:
        """Load comparisons from JSONL file."""
        logger.info(f"Loading comparisons from {self.comparisons_file}")

        with open(self.comparisons_file, 'r', encoding='utf-8') as f:
            self.comparisons = [json.loads(line) for line in f]

        self.reason_texts = [c['reason_text'] for c in self.comparisons]

        logger.info(f"Loaded {len(self.comparisons)} comparisons")

    def analyze_keywords(self) -> Dict[str, int]:
        """Extract and count keywords from reasons."""
        logger.info("Analyzing keywords...")

        # Tokenize and count
        word_counts = Counter()

        for reason in self.reason_texts:
            # Lowercase and split
            words = re.findall(r'\b\w+\b', reason.lower())

            # Filter stop words and very short words
            words = [w for w in words if w not in self.stop_words and len(w) > 2]

            word_counts.update(words)

        logger.info(f"Found {len(word_counts)} unique keywords")

        return dict(word_counts.most_common(100))

    def analyze_phrases(self, n: int = 2) -> Dict[str, int]:
        """
        Extract common n-grams (phrases).

        Args:
            n: N-gram size (2=bigrams, 3=trigrams)

        Returns:
            Dictionary of phrases and counts
        """
        logger.info(f"Analyzing {n}-grams...")

        phrase_counts = Counter()

        for reason in self.reason_texts:
            # Lowercase
            text = reason.lower()

            # Split into words
            words = re.findall(r'\b\w+\b', text)

            # Generate n-grams
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                phrase_counts[ngram] += 1

        # Filter out phrases that are mostly stop words
        filtered = {}
        for phrase, count in phrase_counts.items():
            phrase_words = phrase.split()
            non_stop = [w for w in phrase_words if w not in self.stop_words]

            # Keep if at least half are non-stop words
            if len(non_stop) >= len(phrase_words) / 2:
                filtered[phrase] = count

        logger.info(f"Found {len(filtered)} meaningful {n}-grams")

        return dict(Counter(filtered).most_common(100))

    def categorize_reasons(self) -> Dict[str, List[str]]:
        """
        Categorize reasons based on pre-defined attribute themes.

        Returns:
            Dictionary mapping categories to reason texts
        """
        logger.info("Categorizing reasons by attribute theme...")

        # Category definitions (from existing learned prompts analysis)
        categories = {
            'focus_sharpness': {
                'keywords': ['blur', 'blurry', 'focus', 'focused', 'sharp', 'clear',
                            'detail', 'details', 'hazy', 'fuzzy', 'soft', 'crisp'],
                'reasons': []
            },
            'composition': {
                'keywords': ['crop', 'cropped', 'frame', 'framed', 'framing', 'clutter',
                            'cluttered', 'composition', 'compose', 'composed', 'center',
                            'centered', 'foreground', 'background', 'placement', 'position'],
                'reasons': []
            },
            'exposure_lighting': {
                'keywords': ['dark', 'darker', 'darkest', 'bright', 'brighter', 'brightest',
                            'light', 'lighting', 'lit', 'shadow', 'shadows', 'expose',
                            'exposed', 'exposure', 'overexposed', 'underexposed', 'washed'],
                'reasons': []
            },
            'color_clarity': {
                'keywords': ['color', 'colors', 'colored', 'colorful', 'saturate', 'saturated',
                            'saturation', 'vibrant', 'dull', 'contrast', 'contrasted',
                            'vivid', 'pale', 'faded'],
                'reasons': []
            },
            'content_interest': {
                'keywords': ['boring', 'bored', 'interest', 'interesting', 'interested',
                            'subject', 'random', 'insignificant', 'important', 'notable',
                            'attention', 'engaging', 'dull', 'exciting'],
                'reasons': []
            },
            'view_perspective': {
                'keywords': ['narrow', 'wide', 'view', 'views', 'angle', 'angles',
                            'perspective', 'show', 'shows', 'showing', 'see', 'visible',
                            'far', 'close', 'distance', 'zoom', 'zoomed'],
                'reasons': []
            },
            'specific_elements': {
                'keywords': ['water', 'sky', 'building', 'buildings', 'tree', 'trees',
                            'mountain', 'mountains', 'person', 'people', 'face', 'faces',
                            'car', 'cars', 'boat', 'boats', 'nature', 'landscape'],
                'reasons': []
            }
        }

        # Categorize each reason
        for reason in self.reason_texts:
            reason_lower = reason.lower()

            for category, info in categories.items():
                # Check if any keyword appears in reason
                if any(keyword in reason_lower for keyword in info['keywords']):
                    info['reasons'].append(reason)

        # Log statistics
        for category, info in categories.items():
            count = len(info['reasons'])
            pct = 100.0 * count / len(self.reason_texts)
            logger.info(f"  {category}: {count} reasons ({pct:.1f}%)")

        return {cat: info['reasons'] for cat, info in categories.items()}

    def find_negations(self) -> Dict[str, int]:
        """Find common negation patterns."""
        logger.info("Analyzing negation patterns...")

        negation_patterns = [
            r'too\s+\w+',          # "too dark", "too narrow"
            r'not\s+\w+',          # "not sharp", "not centered"
            r'doesn\'t\s+\w+',     # "doesn't show"
            r'can\'t\s+\w+',       # "can't see"
            r'no\s+\w+',           # "no detail"
            r'without\s+\w+',      # "without focus"
            r'lack\s+\w+',         # "lack detail"
            r'bad\s+\w+',          # "bad lighting"
            r'poor\s+\w+',         # "poor composition"
        ]

        negations = Counter()

        for reason in self.reason_texts:
            reason_lower = reason.lower()

            for pattern in negation_patterns:
                matches = re.findall(pattern, reason_lower)
                negations.update(matches)

        return dict(negations.most_common(50))

    def analyze_length_distribution(self) -> Dict:
        """Analyze reason text length distributions."""
        lengths = [len(r) for r in self.reason_texts]
        word_counts = [len(r.split()) for r in self.reason_texts]

        return {
            'char_lengths': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': sum(lengths) / len(lengths),
                'median': sorted(lengths)[len(lengths) // 2]
            },
            'word_counts': {
                'min': min(word_counts),
                'max': max(word_counts),
                'mean': sum(word_counts) / len(word_counts),
                'median': sorted(word_counts)[len(word_counts) // 2]
            }
        }

    def generate_report(self, output_file: Path) -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")

        # Run all analyses
        keywords = self.analyze_keywords()
        bigrams = self.analyze_phrases(n=2)
        trigrams = self.analyze_phrases(n=3)
        categories = self.categorize_reasons()
        negations = self.find_negations()
        length_stats = self.analyze_length_distribution()

        # Build report
        report = {
            'total_reasons': len(self.reason_texts),
            'unique_reasons': len(set(self.reason_texts)),
            'length_statistics': length_stats,
            'top_keywords': keywords,
            'top_bigrams': bigrams,
            'top_trigrams': trigrams,
            'categories': {
                cat: {
                    'count': len(reasons),
                    'percentage': 100.0 * len(reasons) / len(self.reason_texts),
                    'examples': reasons[:10]  # Sample examples
                }
                for cat, reasons in categories.items()
            },
            'negation_patterns': negations
        }

        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_file}")

        return report

    def print_summary(self, report: Dict) -> None:
        """Print summary of analysis."""
        logger.info("="*70)
        logger.info("REASON TEXT ANALYSIS SUMMARY")
        logger.info("="*70)

        logger.info(f"\nTotal reasons: {report['total_reasons']}")
        logger.info(f"Unique reasons: {report['unique_reasons']}")

        logger.info("\nLength Statistics:")
        logger.info(f"  Characters - Mean: {report['length_statistics']['char_lengths']['mean']:.1f}, "
                   f"Median: {report['length_statistics']['char_lengths']['median']}")
        logger.info(f"  Words - Mean: {report['length_statistics']['word_counts']['mean']:.1f}, "
                   f"Median: {report['length_statistics']['word_counts']['median']}")

        logger.info("\nTop 20 Keywords:")
        for i, (word, count) in enumerate(list(report['top_keywords'].items())[:20], 1):
            logger.info(f"  {i:2d}. {word:20s} {count:5d}")

        logger.info("\nTop 15 Bigrams:")
        for i, (phrase, count) in enumerate(list(report['top_bigrams'].items())[:15], 1):
            logger.info(f"  {i:2d}. {phrase:30s} {count:5d}")

        logger.info("\nCategory Distribution:")
        for cat, info in report['categories'].items():
            logger.info(f"  {cat:25s} {info['count']:5d} ({info['percentage']:5.1f}%)")

        logger.info("\nTop 10 Negation Patterns:")
        for i, (neg, count) in enumerate(list(report['negation_patterns'].items())[:10], 1):
            logger.info(f"  {i:2d}. {neg:25s} {count:5d}")

        logger.info("="*70)


def main():
    """Main analysis pipeline."""
    # Paths
    comparisons_file = Path("data/phototriage/raw_comparisons.jsonl")
    output_file = Path("data/phototriage/reason_analysis.json")

    # Check if input exists
    if not comparisons_file.exists():
        logger.error(f"Comparisons file not found: {comparisons_file}")
        logger.error("Please run 01_extract_reasons.py first")
        return

    # Create analyzer
    logger.info("="*70)
    logger.info("PhotoTriage Reason Analysis")
    logger.info("="*70)

    analyzer = ReasonAnalyzer(comparisons_file)

    # Load comparisons
    analyzer.load_comparisons()

    # Generate report
    report = analyzer.generate_report(output_file)

    # Print summary
    analyzer.print_summary(report)


if __name__ == "__main__":
    main()
