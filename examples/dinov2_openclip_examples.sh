#!/bin/bash
# Example commands for using DINOv2 and OpenCLIP methods

# ============================================================
# DINOv2 Examples
# ============================================================

# Basic DINOv2 with default (base) model
python -m sim_bench.cli --methods dinov2 --datasets ukbench

# Quick test with DINOv2
python -m sim_bench.cli --quick --quick-size 50 --methods dinov2 --datasets ukbench

# DINOv2 with different model sizes (edit configs/methods/dinov2.yaml first)
# variant: "small"   # Fastest, 384-dim
# variant: "base"    # Default, 768-dim
# variant: "large"   # Best quality, 1024-dim
# variant: "giant"   # Highest quality, 1536-dim
python -m sim_bench.cli --methods dinov2 --datasets holidays

# DINOv2 on multiple datasets
python -m sim_bench.cli --methods dinov2 --datasets ukbench,holidays

# ============================================================
# OpenCLIP Examples
# ============================================================

# Basic OpenCLIP with default (ViT-B-32) model
python -m sim_bench.cli --methods openclip --datasets ukbench

# Quick test with OpenCLIP
python -m sim_bench.cli --quick --quick-size 50 --methods openclip --datasets holidays

# OpenCLIP with different models (edit configs/methods/openclip.yaml first)
# Fast models:
#   model: "ViT-B-32", pretrained: "laion2b_s34b_b79k"
#   model: "ViT-B-16", pretrained: "laion2b_s34b_b88k"
# High-quality models:
#   model: "ViT-L-14", pretrained: "laion2b_s32b_b82k"
#   model: "ViT-H-14", pretrained: "laion2b_s32b_b79k"
python -m sim_bench.cli --methods openclip --datasets ukbench

# ============================================================
# Comparative Benchmarks
# ============================================================

# Compare all deep learning methods
python -m sim_bench.cli --methods deep,dinov2,openclip --datasets ukbench

# Compare DINOv2 vs OpenCLIP
python -m sim_bench.cli --methods dinov2,openclip --datasets holidays

# Compare all methods (traditional + deep learning)
python -m sim_bench.cli --methods chi_square,emd,deep,dinov2,openclip --datasets ukbench

# Full benchmark on both datasets
python -m sim_bench.cli --methods dinov2,openclip --datasets ukbench,holidays

# ============================================================
# Performance Optimization
# ============================================================

# With GPU acceleration (edit config: device: "cuda")
python -m sim_bench.cli --methods dinov2 --datasets ukbench

# With larger batch size for GPU (edit config: batch_size: 32)
python -m sim_bench.cli --methods openclip --datasets holidays

# Disable caching for testing
python -m sim_bench.cli --no-cache --methods dinov2 --datasets ukbench

# ============================================================
# Analysis
# ============================================================

# After running experiments, analyze results in Jupyter
# jupyter notebook sim_bench/analysis/methods_comparison.ipynb

# Or use the analysis notebook at project root
# jupyter notebook analyze_results.ipynb



