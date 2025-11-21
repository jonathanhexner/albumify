#!/usr/bin/env python
"""
Simple explanation of how CLIP aesthetic scoring works.

Shows step-by-step how multiple prompts become a single score.
"""

def explain_scoring():
    """Explain the scoring process with a concrete example."""
    
    print("="*80)
    print("CLIP Aesthetic Scoring: From Multiple Prompts to One Score")
    print("="*80)
    
    print("\nIMAGE A")
    print("-" * 80)
    print("\nStep 1: CLIP encodes image and all prompts")
    print("  Image A -> embedding vector (512 dimensions)")
    print("  9 contrastive pairs (18 prompts) -> 18 embedding vectors")
    
    print("\nStep 2: Compute similarity between image and each prompt")
    print("  (Cosine similarity, range: -1 to 1)")
    print("\n  Contrastive Pair 1:")
    print("    'sharp and well-focused'     -> similarity: 0.65")
    print("    'blurry and out-of-focus'    -> similarity: 0.20")
    print("    Contrastive score: 0.65 - 0.20 = 0.45")
    
    print("\n  Contrastive Pair 2:")
    print("    'well-composed and uncluttered' -> similarity: 0.70")
    print("    'cluttered and poorly-composed' -> similarity: 0.15")
    print("    Contrastive score: 0.70 - 0.15 = 0.55")
    
    print("\n  ... (7 more pairs)")
    print("    Average of all 9 contrastive scores = 0.50")
    
    print("\nStep 3: Aggregate into final score")
    print("  Using 'weighted' aggregation:")
    print("    final_score = 0.5 * contrastive + 0.3 * positive + 0.2 * negative")
    print("    final_score = 0.5 * 0.50 + 0.3 * 0 + 0.2 * 0")
    print("    final_score = 0.25")
    
    print("\n" + "="*80)
    print("RESULT: Image A gets score_a = 0.25")
    print("="*80)
    
    print("\nIMAGE B")
    print("-" * 80)
    print("(Same process, different similarities)")
    print("  Result: score_b = 0.15")
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print("  score_a (0.25) > score_b (0.15)")
    print("  -> Prediction: Image A is better quality")
    print("  -> If ground truth says A was preferred: [CORRECT]")
    print("  -> If ground truth says B was preferred: [WRONG]")
    
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("Even though we use 9-18 different text prompts,")
    print("they are ALL combined into a SINGLE number (score_a or score_b).")
    print("\nThe multiple prompts capture different quality aspects:")
    print("  - Sharpness, composition, framing, exposure, color, etc.")
    print("But the final output is just one score per image.")
    print("\nFor pairwise comparison, we simply compare:")
    print("  score_a vs score_b")


if __name__ == '__main__':
    explain_scoring()

