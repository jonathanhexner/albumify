#!/usr/bin/env python
"""
Show text prompts used by each CLIP aesthetic variant.

This helps understand what prompts were used in CLIP runs,
especially for existing results that may not have prompt info saved.
"""

import yaml
from pathlib import Path


def show_default_prompts():
    """Show default prompts used by laion/sac variants."""
    print("="*80)
    print("Default CLIP Aesthetic Prompts (laion/sac variants)")
    print("="*80)
    
    contrastive = [
        ("a well-composed photograph", "a poorly-composed photograph"),
        ("a photo with the subject well placed in the frame",
         "a photo with the subject not well placed in the frame"),
        ("a photo that is well cropped", "a photo that is poorly cropped"),
        ("Good Quality photo", "Poor Quality photo"),
    ]
    
    positive = [
        "professional photography",
        "aesthetically pleasing",
    ]
    
    negative = [
        "amateur snapshot",
        "poor framing",
    ]
    
    print("\nContrastive Pairs (4 pairs):")
    for i, (pos, neg) in enumerate(contrastive, 1):
        print(f"  {i}. +: {pos}")
        print(f"     -: {neg}")
    
    print(f"\nPositive Attributes ({len(positive)}):")
    for attr in positive:
        print(f"  - {attr}")
    
    print(f"\nNegative Attributes ({len(negative)}):")
    for attr in negative:
        print(f"  - {attr}")
    
    print(f"\nTotal prompts: {len(contrastive)*2 + len(positive) + len(negative)}")


def show_learned_prompts(prompts_file: str = "configs/learned_aesthetic_prompts.yaml"):
    """Show learned prompts from YAML file."""
    prompts_path = Path(prompts_file)
    
    if not prompts_path.exists():
        print(f"Warning: Learned prompts file not found: {prompts_file}")
        return
    
    print("\n" + "="*80)
    print("Learned CLIP Aesthetic Prompts (learned variant)")
    print(f"Source: {prompts_file}")
    print("="*80)
    
    with open(prompts_path, 'r') as f:
        prompts_data = yaml.safe_load(f)
    
    contrastive = prompts_data.get('contrastive_pairs', [])
    
    print(f"\nContrastive Pairs ({len(contrastive)} pairs):")
    for i, pair in enumerate(contrastive, 1):
        pos, neg = pair
        print(f"  {i}. +: {pos}")
        print(f"     -: {neg}")
    
    print(f"\nTotal prompts: {len(contrastive)*2}")


def show_variant_summary():
    """Show summary of all variants."""
    print("\n" + "="*80)
    print("CLIP Aesthetic Variants Summary")
    print("="*80)
    
    variants = {
        'laion': {
            'pretrained': 'laion2b_s34b_b79k',
            'prompts': 'Default prompts (see above)',
            'description': 'LAION-2B pretrained model with default aesthetic prompts'
        },
        'sac': {
            'pretrained': 'laion2b_s34b_b79k',
            'prompts': 'Default prompts (same as laion)',
            'description': 'Same as laion (fixed: uses valid checkpoint for ViT-B-32)'
        },
        'learned': {
            'pretrained': 'laion2b_s34b_b79k',
            'prompts': 'Learned from PhotoTriage dataset (see above)',
            'description': 'Prompts learned from user preferences in PhotoTriage'
        }
    }
    
    for variant, info in variants.items():
        print(f"\n{variant.upper()}:")
        print(f"  Pretrained: {info['pretrained']}")
        print(f"  Prompts: {info['prompts']}")
        print(f"  Description: {info['description']}")


def main():
    """Main entry point."""
    show_default_prompts()
    show_learned_prompts()
    show_variant_summary()
    
    print("\n" + "="*80)
    print("Note: For existing benchmark results, check summary.json for")
    print("      saved prompt information (if run with updated code).")
    print("="*80)


if __name__ == '__main__':
    main()

