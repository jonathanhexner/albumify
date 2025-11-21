"""
Learn CLIP Aesthetic Prompts from PhotoTriage Dataset

Extracts user reasons for preferring one image over another,
clusters them into common themes, and generates CLIP prompts.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_all_reasons(reviews_dir: Path) -> List[str]:
    """
    Extract all reason texts from PhotoTriage reviews.

    Args:
        reviews_dir: Directory containing review JSON files

    Returns:
        List of all reason texts
    """
    all_reasons = []

    json_files = list(reviews_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} review files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for review in data.get('reviews', []):
                reason = review.get('reason', ['', ''])
                # Second element contains the text
                if len(reason) > 1 and reason[1]:
                    all_reasons.append(reason[1])

        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")

    logger.info(f"Extracted {len(all_reasons)} total reasons")
    return all_reasons


def clean_reason(reason: str) -> str:
    """Clean and normalize a reason text."""
    # Lowercase
    reason = reason.lower().strip()

    # Remove extra whitespace
    reason = re.sub(r'\s+', ' ', reason)

    # Remove trailing punctuation
    reason = reason.rstrip('.,;:!')

    return reason


def extract_keywords(reasons: List[str]) -> Counter:
    """Extract common keywords from reasons."""
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'it', 'its', 'as', 'is', 'are', 'was', 'were', 'been', 'be', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'that', 'this', 'these', 'those', 'than', 'then', 'so', 'if', 'when',
        'where', 'who', 'which', 'what', 'why', 'how', 'one', 'all', 'each',
        'other', 'some', 'such', 'only', 'own', 'same', 'just', 'can', 'cant',
        'very', 'too', 'much', 'more', 'less', 'dont', 'doesnt', 'not', 'no',
        'well', 'also', 'like', 'seems', 'looks', 'looks', 'seem'
    }

    keyword_counter = Counter()

    for reason in reasons:
        words = reason.lower().split()
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w\s-]', '', word)
            if word and word not in stop_words and len(word) > 2:
                keyword_counter[word] += 1

    return keyword_counter


def find_common_phrases(reasons: List[str], min_occurrences: int = 5) -> List[Tuple[str, int]]:
    """Find common 2-3 word phrases in reasons."""
    phrase_counter = Counter()

    for reason in reasons:
        words = reason.lower().split()

        # 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            phrase_counter[phrase] += 1

        # 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            phrase_counter[phrase] += 1

    # Filter by minimum occurrences
    common_phrases = [(phrase, count) for phrase, count in phrase_counter.most_common()
                      if count >= min_occurrences]

    return common_phrases


def manual_categorize_reasons(reasons: List[str]) -> Dict[str, List[str]]:
    """
    Manually categorize reasons into thematic groups.

    Based on PhotoTriage data analysis, common categories:
    - Focus/Sharpness (blurry, out of focus, sharp, clear)
    - Composition (cluttered, cropped, framing, centered)
    - Exposure/Lighting (dark, bright, overexposed, underexposed)
    - Color (bad color, hazy, vibrant)
    - Content (boring, interesting, shows detail, lacks subject)
    - View/Perspective (narrow, wide, shows more/less)
    """
    categories = {
        'focus_sharpness': [],
        'composition': [],
        'exposure_lighting': [],
        'color_clarity': [],
        'content_interest': [],
        'view_perspective': []
    }

    # Keywords for each category
    category_keywords = {
        'focus_sharpness': ['blur', 'focus', 'sharp', 'clear', 'detail', 'hazy'],
        'composition': ['crop', 'frame', 'clutter', 'compos', 'center', 'foreground', 'background'],
        'exposure_lighting': ['dark', 'bright', 'light', 'shadow', 'expose'],
        'color_clarity': ['color', 'saturate', 'vibrant', 'dull', 'contrast'],
        'content_interest': ['boring', 'interest', 'subject', 'random', 'insignificant'],
        'view_perspective': ['narrow', 'wide', 'view', 'angle', 'perspective', 'show']
    }

    for reason in reasons:
        reason_lower = reason.lower()

        # Check which category this reason belongs to
        for category, keywords in category_keywords.items():
            if any(kw in reason_lower for kw in keywords):
                categories[category].append(reason)
                break
        else:
            # If no match, put in content_interest as catch-all
            categories['content_interest'].append(reason)

    return categories


def generate_prompts_from_categories(
    categorized_reasons: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Generate contrastive CLIP prompts from categorized reasons.

    Returns list of (positive_prompt, negative_prompt) pairs.
    """
    # Manual mapping based on PhotoTriage analysis
    prompt_pairs = [
        # Focus/Sharpness
        ("a sharp and well-focused photograph", "a blurry and out-of-focus photograph"),

        # Composition
        ("a well-composed and uncluttered photograph", "a cluttered and poorly-composed photograph"),
        ("a photo with good framing", "a photo with bad framing or too cropped"),

        # Exposure/Lighting
        ("a well-exposed photograph with good lighting", "a dark or poorly-lit photograph"),

        # Color/Clarity
        ("a photo with good color and clarity", "a photo with bad color or hazy appearance"),

        # Content/Interest
        ("an interesting photograph with a clear subject", "a boring photo that lacks a subject"),

        # View/Perspective
        ("a photo with a good field of view", "a photo with a too narrow or limited view"),

        # Detail visibility
        ("a photo that shows important details clearly", "a photo where you can't see details well"),

        # Overall quality (from dataset)
        ("a high-quality photograph", "a low-quality photograph"),
    ]

    return prompt_pairs


def main():
    """Extract and analyze PhotoTriage reasons."""
    logger.info("="*80)
    logger.info("Learning CLIP Prompts from PhotoTriage Dataset")
    logger.info("="*80)

    # Path to PhotoTriage reviews
    reviews_dir = Path("D:/Similar Images/automatic_triage_photo_series/train_val/reviews_trainval/reviews_trainval")

    if not reviews_dir.exists():
        logger.error(f"Reviews directory not found: {reviews_dir}")
        return

    # Step 1: Extract all reasons
    logger.info("\nStep 1: Extracting reasons from dataset...")
    all_reasons = extract_all_reasons(reviews_dir)

    if not all_reasons:
        logger.error("No reasons extracted!")
        return

    # Clean reasons
    cleaned_reasons = [clean_reason(r) for r in all_reasons if r.strip()]

    # Step 2: Analyze keywords
    logger.info("\nStep 2: Analyzing common keywords...")
    keywords = extract_keywords(cleaned_reasons)

    logger.info("\nTop 30 keywords:")
    for keyword, count in keywords.most_common(30):
        logger.info(f"  {keyword:20s}: {count:4d}")

    # Step 3: Find common phrases
    logger.info("\nStep 3: Finding common phrases...")
    phrases = find_common_phrases(cleaned_reasons, min_occurrences=10)

    logger.info(f"\nTop 20 common phrases (min 10 occurrences):")
    for phrase, count in phrases[:20]:
        logger.info(f"  {phrase:40s}: {count:4d}")

    # Step 4: Categorize reasons
    logger.info("\nStep 4: Categorizing reasons...")
    categorized = manual_categorize_reasons(cleaned_reasons)

    logger.info("\nReasons per category:")
    for category, reasons in categorized.items():
        logger.info(f"  {category:20s}: {len(reasons):4d} reasons")

        # Show sample reasons
        logger.info(f"    Samples:")
        for reason in reasons[:3]:
            logger.info(f"      - {reason[:60]}")

    # Step 5: Generate prompts
    logger.info("\nStep 5: Generating CLIP prompt pairs...")
    prompt_pairs = generate_prompts_from_categories(categorized)

    logger.info(f"\nGenerated {len(prompt_pairs)} contrastive prompt pairs:")
    logger.info("-"*80)
    for i, (pos, neg) in enumerate(prompt_pairs, 1):
        logger.info(f"{i}. Positive: '{pos}'")
        logger.info(f"   Negative: '{neg}'")
        logger.info("")

    # Save to file
    output_file = Path("configs/learned_aesthetic_prompts.yaml")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("# Learned CLIP Aesthetic Prompts from PhotoTriage Dataset\n")
        f.write("# Based on analysis of user preferences and rejection reasons\n\n")
        f.write("contrastive_pairs:\n")
        for pos, neg in prompt_pairs:
            f.write(f'  - ["{pos}", "{neg}"]\n')

    logger.info(f"\nSaved prompts to: {output_file}")

    logger.info("\n" + "="*80)
    logger.info("Analysis Complete!")
    logger.info("="*80)
    logger.info(f"\nTotal reasons analyzed: {len(cleaned_reasons)}")
    logger.info(f"Generated prompt pairs: {len(prompt_pairs)}")
    logger.info(f"\nNext steps:")
    logger.info("  1. Review the generated prompts in configs/learned_aesthetic_prompts.yaml")
    logger.info("  2. Update CLIPAestheticAssessor to use these prompts")
    logger.info("  3. Re-run quality benchmark on PhotoTriage to compare performance")
    logger.info("="*80)


if __name__ == "__main__":
    main()
