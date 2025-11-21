"""
Factory function for creating photo analyzers.
"""

from sim_bench.photo_analysis.clip_tagger import CLIPTagger


def create_photo_analyzer(analyzer_type: str, **config):
    """
    Factory function for creating photo analyzers.

    Args:
        analyzer_type: Type of analyzer ('clip', 'blip', etc.)
        **config: Analyzer-specific configuration

    Returns:
        PhotoAnalyzer instance

    Raises:
        ValueError: If unknown analyzer type

    Example:
        >>> from sim_bench.photo_analysis.factory import create_photo_analyzer
        >>>
        >>> tagger = create_photo_analyzer('clip', device='cuda')
        >>> analysis = tagger.analyze_image('photo.jpg')
        >>> print(analysis['primary_tags'])
    """
    if analyzer_type == 'clip':
        return CLIPTagger(**config)
    else:
        raise ValueError(
            f"Unknown photo analyzer type: {analyzer_type}. "
            f"Supported types: 'clip'"
        )
