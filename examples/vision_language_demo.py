"""
Vision-Language Model Demo

Demonstrates the new sim_bench.vision_language module capabilities:
1. Image encoding
2. Text encoding
3. Image-text similarity
4. Aesthetic assessment
5. Semantic retrieval
6. Zero-shot classification
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_basic_encoding():
    """Demo 1: Basic image and text encoding."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Image and Text Encoding")
    print("="*80)

    from sim_bench.vision_language import CLIPModel

    # Create CLIP model
    clip = CLIPModel("ViT-B-32", device="cpu")
    print(f"Model: {clip}")
    print(f"Embedding dim: {clip.get_embedding_dim()}")

    # Find sample images
    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("No sample images found, skipping")
        return

    sample_images = list(samples_dir.glob("*.jpg"))[:3]
    if not sample_images:
        print("No sample images found, skipping")
        return

    print(f"\nEncoding {len(sample_images)} images...")
    image_embs = clip.encode_images([str(p) for p in sample_images])
    print(f"Image embeddings shape: {image_embs.shape}")

    print("\nEncoding text prompts...")
    texts = ["a photograph", "outdoor scene", "indoor scene"]
    text_embs = clip.encode_texts(texts)
    print(f"Text embeddings shape: {text_embs.shape}")

    # Compute similarities
    print("\nImage-text similarity matrix:")
    similarities = clip.compute_similarity(image_embs, text_embs)
    print(similarities)

    print("\n[OK] Basic encoding demo complete")


def demo_aesthetic_assessment():
    """Demo 2: Aesthetic quality assessment."""
    print("\n" + "="*80)
    print("DEMO 2: Aesthetic Quality Assessment")
    print("="*80)

    from sim_bench.vision_language import CLIPModel
    from sim_bench.vision_language.applications import AestheticAssessor

    # Create CLIP model
    clip = CLIPModel("ViT-B-32", device="cpu")

    # Create aesthetic assessor
    assessor = AestheticAssessor(clip, aggregation="weighted")
    print(f"Aesthetic assessor created")
    print(f"Prompts: {assessor.get_prompt_summary()}")

    # Find sample images
    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("No sample images found, skipping")
        return

    sample_images = list(samples_dir.glob("*.jpg"))[:5]
    if not sample_images:
        print("No sample images found, skipping")
        return

    print(f"\nAssessing {len(sample_images)} images...")
    for img_path in sample_images:
        score, detailed = assessor.assess_with_details(str(img_path))
        print(f"\n{img_path.name}:")
        print(f"  Overall score: {score:.4f}")

        # Show top contrastive scores
        contrastive = {k: v for k, v in detailed.items() if k.startswith('contrast_')}
        top_3 = sorted(contrastive.items(), key=lambda x: x[1], reverse=True)[:3]
        for key, val in top_3:
            print(f"    {key[:50]}: {val:.4f}")

    print("\n[OK] Aesthetic assessment demo complete")


def demo_semantic_retrieval():
    """Demo 3: Semantic image retrieval."""
    print("\n" + "="*80)
    print("DEMO 3: Semantic Image Retrieval")
    print("="*80)

    from sim_bench.vision_language import CLIPModel
    from sim_bench.vision_language.applications import SemanticRetrieval

    # Create CLIP model
    clip = CLIPModel("ViT-B-32", device="cpu")

    # Create retrieval system
    retrieval = SemanticRetrieval(clip)

    # Find sample images
    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("No sample images found, skipping")
        return

    sample_images = [str(p) for p in samples_dir.glob("*.jpg")]
    if not sample_images:
        print("No sample images found, skipping")
        return

    print(f"Indexing {len(sample_images)} images...")
    retrieval.index_images(sample_images, verbose=False)

    # Search queries
    queries = [
        "outdoor scene",
        "people in photo",
        "building or architecture"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = retrieval.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {Path(result['path']).name} (score: {result['score']:.4f})")

    print("\n[OK] Semantic retrieval demo complete")


def demo_zero_shot_classification():
    """Demo 4: Zero-shot classification."""
    print("\n" + "="*80)
    print("DEMO 4: Zero-Shot Classification")
    print("="*80)

    from sim_bench.vision_language import CLIPModel
    from sim_bench.vision_language.applications import ZeroShotClassifier

    # Create CLIP model
    clip = CLIPModel("ViT-B-32", device="cpu")

    # Create classifier
    classifier = ZeroShotClassifier(clip, temperature=1.0)

    # Define classes
    classes = {
        "outdoor": "a photo taken outdoors",
        "indoor": "a photo taken indoors",
        "people": "a photo with people in it",
        "objects": "a photo of objects or things"
    }

    print(f"Classes: {list(classes.keys())}")

    # Find sample images
    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("No sample images found, skipping")
        return

    sample_images = [str(p) for p in samples_dir.glob("*.jpg")][:5]
    if not sample_images:
        print("No sample images found, skipping")
        return

    print(f"\nClassifying {len(sample_images)} images...")
    for img_path in sample_images:
        result = classifier.classify(img_path, classes, return_probs=True)
        print(f"\n{Path(img_path).name}:")
        print(f"  Predicted: {result['class_name']} ({result['confidence']:.1%})")
        print(f"  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls:10s}: {prob:.1%}")

    print("\n[OK] Zero-shot classification demo complete")


def demo_unified_workflow():
    """Demo 5: Using same CLIP model for multiple tasks."""
    print("\n" + "="*80)
    print("DEMO 5: Unified Workflow - One Model, Multiple Tasks")
    print("="*80)

    from sim_bench.vision_language import CLIPModel
    from sim_bench.vision_language.applications import (
        AestheticAssessor,
        SemanticRetrieval,
        ZeroShotClassifier
    )

    # Create ONE CLIP model
    print("Creating single CLIP model...")
    clip = CLIPModel("ViT-B-32", device="cpu")

    # Use for multiple tasks
    aesthetic = AestheticAssessor(clip)
    retrieval = SemanticRetrieval(clip)
    classifier = ZeroShotClassifier(clip)

    print(f"[OK] Created 3 applications from 1 CLIP model")
    print(f"[OK] Shared cache: {clip.get_cache_stats()}")

    # Find sample images
    samples_dir = Path("samples/ukbench")
    if not samples_dir.exists():
        print("No sample images found, skipping")
        return

    sample_images = [str(p) for p in samples_dir.glob("*.jpg")][:3]
    if not sample_images:
        print("No sample images found, skipping")
        return

    test_image = sample_images[0]
    print(f"\nAnalyzing: {Path(test_image).name}")

    # 1. Assess quality
    quality_score = aesthetic.assess_image(test_image)
    print(f"  Quality score: {quality_score:.4f}")

    # 2. Classify
    classes = {"outdoor": "outdoor scene", "indoor": "indoor scene"}
    classification = classifier.classify(test_image, classes)
    print(f"  Classification: {classification['class_name']} ({classification['confidence']:.1%})")

    # 3. Find similar
    retrieval.index_images(sample_images, verbose=False)
    similar = retrieval.get_similar_images(test_image, top_k=2)
    print(f"  Similar images:")
    for result in similar:
        print(f"    - {Path(result['path']).name} (score: {result['score']:.3f})")

    print("\n[OK] Unified workflow demo complete")
    print(f"[OK] Final cache stats: {clip.get_cache_stats()}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("VISION-LANGUAGE MODEL DEMONSTRATIONS")
    print("="*80)
    print("\nDemonstrating the new sim_bench.vision_language module.")
    print("This module provides unified API for vision-language models.")

    try:
        demo_basic_encoding()
    except Exception as e:
        print(f"\n[ERROR] Demo 1 failed: {e}")

    try:
        demo_aesthetic_assessment()
    except Exception as e:
        print(f"\n[ERROR] Demo 2 failed: {e}")

    try:
        demo_semantic_retrieval()
    except Exception as e:
        print(f"\n[ERROR] Demo 3 failed: {e}")

    try:
        demo_zero_shot_classification()
    except Exception as e:
        print(f"\n[ERROR] Demo 4 failed: {e}")

    try:
        demo_unified_workflow()
    except Exception as e:
        print(f"\n[ERROR] Demo 5 failed: {e}")

    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE")
    print("="*80)
    print("\nKey benefits of vision_language module:")
    print("  1. Unified API for all VL models (CLIP, BLIP, etc.)")
    print("  2. Reusable models across multiple tasks")
    print("  3. No code duplication")
    print("  4. Easy to extend with new models and applications")
    print("  5. Backward compatible with existing code")
    print("="*80)


if __name__ == "__main__":
    main()
