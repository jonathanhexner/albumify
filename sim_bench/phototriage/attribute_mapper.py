"""
Attribute mapping from reason texts to quality attributes.

Maps user-provided reason texts to structured attribute labels
using keyword matching and natural language understanding.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AttributeCategory(Enum):
    """High-level attribute categories."""
    FOCUS_CLARITY = "focus_clarity"
    COMPOSITION = "composition"
    EXPOSURE_LIGHTING = "exposure_lighting"
    PERSPECTIVE = "perspective"
    CONTENT = "content"


@dataclass
class AttributeLabel:
    """A single attribute label extracted from a reason."""
    name: str
    winner: str  # "A" or "B"
    confidence: float  # 0.0-1.0
    reason_snippet: str  # Part of reason that triggered this attribute
    category: AttributeCategory


class AttributeMapper:
    """
    Map reason texts to attribute labels.

    Uses a combination of:
    - Keyword matching
    - Phrase pattern matching
    - Negation handling
    - Context-aware rules
    """

    def __init__(self):
        """Initialize attribute mapper with attribute definitions."""
        self._init_attribute_patterns()

    def _init_attribute_patterns(self) -> None:
        """Initialize keyword patterns for each attribute."""

        # Define attribute patterns
        # Format: attribute_name -> (keywords, negation_keywords, category)
        self.attributes = {
            # Focus & Clarity
            'sharpness': (
                ['blur', 'blurry', 'focus', 'focused', 'sharp', 'clear',
                 'fuzzy', 'soft', 'crisp', 'definition'],
                ['not blur', 'not blurry', 'not focused', 'out of focus',
                 'too blur', 'very blur'],
                AttributeCategory.FOCUS_CLARITY
            ),
            'detail_visibility': (
                ['detail', 'details', 'see', 'visible', 'visibility', 'hazy',
                 'can see', 'can\'t see', 'shows', 'doesn\'t show'],
                ['can\'t see', 'cannot see', 'no detail', 'lack detail',
                 'doesn\'t show', 'hard to see'],
                AttributeCategory.FOCUS_CLARITY
            ),
            'motion_blur': (
                ['motion', 'movement', 'moving', 'static', 'still'],
                ['motion blur', 'moving'],
                AttributeCategory.FOCUS_CLARITY
            ),

            # Composition & Framing
            'framing': (
                ['frame', 'framed', 'framing', 'boundary', 'boundaries',
                 'edge', 'edges'],
                ['bad frame', 'poor frame', 'not framed', 'badly framed'],
                AttributeCategory.COMPOSITION
            ),
            'cropping_completeness': (
                ['crop', 'cropped', 'cut', 'cut off', 'incomplete', 'complete',
                 'whole', 'full', 'partial', 'truncated'],
                ['cropped', 'cut off', 'not complete', 'incomplete'],
                AttributeCategory.COMPOSITION
            ),
            'subject_placement': (
                ['center', 'centered', 'position', 'positioned', 'placement',
                 'located', 'off-center', 'middle'],
                ['not centered', 'off-center', 'bad position', 'poorly positioned'],
                AttributeCategory.COMPOSITION
            ),
            'background_clutter': (
                ['clutter', 'cluttered', 'messy', 'busy', 'clean', 'clear',
                 'distracting', 'distraction', 'background', 'foreground'],
                ['cluttered', 'messy', 'busy background', 'distracting'],
                AttributeCategory.COMPOSITION
            ),

            # Exposure & Lighting
            'exposure_quality': (
                ['dark', 'darker', 'bright', 'brighter', 'expose', 'exposed',
                 'exposure', 'lighting', 'overexposed', 'underexposed', 'washed'],
                ['too dark', 'too bright', 'overexposed', 'underexposed',
                 'poorly lit', 'bad lighting'],
                AttributeCategory.EXPOSURE_LIGHTING
            ),
            'lighting_quality': (
                ['light', 'lighting', 'lit', 'shadow', 'shadows', 'illuminate',
                 'illuminated', 'brightness'],
                ['harsh', 'harsh light', 'flat lighting', 'bad shadow',
                 'poor lighting'],
                AttributeCategory.EXPOSURE_LIGHTING
            ),
            'dynamic_range': (
                ['contrast', 'washed', 'blown', 'crushed', 'highlight', 'highlights',
                 'shadow detail'],
                ['washed out', 'blown out', 'crushed', 'no contrast', 'flat'],
                AttributeCategory.EXPOSURE_LIGHTING
            ),

            # Perspective & Field of View
            'field_of_view': (
                ['narrow', 'wide', 'view', 'field', 'panorama', 'panoramic',
                 'shows more', 'shows less', 'limited view'],
                ['too narrow', 'narrow view', 'limited view', 'doesn\'t show'],
                AttributeCategory.PERSPECTIVE
            ),
            'distance_appropriateness': (
                ['far', 'close', 'distance', 'zoom', 'zoomed', 'near', 'away',
                 'closer', 'farther'],
                ['too far', 'too close', 'far away', 'very far'],
                AttributeCategory.PERSPECTIVE
            ),

            # Content & Interest
            'subject_interest': (
                ['boring', 'interesting', 'interest', 'engaging', 'dull',
                 'exciting', 'captivating', 'subject', 'attention'],
                ['boring', 'dull', 'not interesting', 'lack interest'],
                AttributeCategory.CONTENT
            ),
        }

        # Compile regex patterns for efficiency
        self.keyword_patterns = {}
        self.negation_patterns = {}

        for attr, (keywords, negations, category) in self.attributes.items():
            # Create word boundary patterns for keywords
            keyword_regex = '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in keywords)
            self.keyword_patterns[attr] = re.compile(keyword_regex, re.IGNORECASE)

            # Create patterns for negations
            negation_regex = '|'.join(re.escape(neg) for neg in negations)
            self.negation_patterns[attr] = re.compile(negation_regex, re.IGNORECASE)

    def map_reason_to_attributes(
        self,
        reason_text: str,
        user_choice: str  # "LEFT" or "RIGHT"
    ) -> List[AttributeLabel]:
        """
        Map a reason text to attribute labels.

        Args:
            reason_text: The user's reason text
            user_choice: Which image was preferred ("LEFT" or "RIGHT")

        Returns:
            List of AttributeLabel objects
        """
        attributes = []

        # Normalize reason text
        reason_lower = reason_text.lower().strip()

        # Check each attribute
        for attr_name, (keywords, negations, category) in self.attributes.items():
            # Check if attribute is mentioned
            keyword_match = self.keyword_patterns[attr_name].search(reason_lower)

            if not keyword_match:
                continue

            # Check for negation patterns
            negation_match = self.negation_patterns[attr_name].search(reason_lower)

            # Determine winner and confidence
            winner, confidence, snippet = self._determine_winner(
                reason_text,
                reason_lower,
                user_choice,
                keyword_match,
                negation_match
            )

            # Create attribute label
            label = AttributeLabel(
                name=attr_name,
                winner=winner,
                confidence=confidence,
                reason_snippet=snippet,
                category=category
            )

            attributes.append(label)

        return attributes

    def _determine_winner(
        self,
        reason_text: str,
        reason_lower: str,
        user_choice: str,
        keyword_match,
        negation_match
    ) -> Tuple[str, float, str]:
        """
        Determine which image wins for the attribute.

        Logic:
        - If negation found (e.g., "too dark", "blurry"):
          - This describes the LOSING image
          - Winner is the opposite of user_choice
        - If positive mention (e.g., "sharp", "clear"):
          - This describes the WINNING image
          - Winner is user_choice

        Returns:
            (winner, confidence, snippet)
        """
        # Extract matched snippet
        match_start = keyword_match.start()
        match_end = keyword_match.end()

        # Get context around match (±30 chars)
        snippet_start = max(0, match_start - 30)
        snippet_end = min(len(reason_text), match_end + 30)
        snippet = reason_text[snippet_start:snippet_end].strip()

        # Default confidence
        confidence = 0.8

        # Map user choice to winner/loser
        if user_choice == "LEFT":
            preferred = "A"
            rejected = "B"
        elif user_choice == "RIGHT":
            preferred = "B"
            rejected = "A"
        else:
            # Unknown choice
            return "A", 0.5, snippet

        # Check for negation
        if negation_match:
            # Negation found - reason describes the LOSING image
            winner = preferred  # Preferred image is better
            confidence = 0.9  # High confidence for negations
        else:
            # Check for positive indicators
            positive_patterns = [
                r'\bgood\b', r'\bbetter\b', r'\bnice\b', r'\bwell\b',
                r'\bclear\b', r'\bsharp\b', r'\bbright\b', r'\bclean\b'
            ]

            has_positive = any(re.search(pat, reason_lower) for pat in positive_patterns)

            if has_positive:
                # Positive mention - describes winning image
                winner = preferred
                confidence = 0.85
            else:
                # Ambiguous - could describe either
                # Default: assume it describes the issue with losing image
                winner = preferred
                confidence = 0.7

        return winner, confidence, snippet

    def get_attribute_schema(self) -> Dict:
        """Get the complete attribute schema."""
        return {
            attr: {
                'keywords': keywords,
                'negations': negations,
                'category': category.value
            }
            for attr, (keywords, negations, category) in self.attributes.items()
        }

    def get_attribute_names(self) -> List[str]:
        """Get list of all attribute names."""
        return list(self.attributes.keys())

    def get_attributes_by_category(self) -> Dict[AttributeCategory, List[str]]:
        """Get attributes grouped by category."""
        by_category = {}

        for attr, (_, _, category) in self.attributes.items():
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(attr)

        return by_category
