# Dependency Management Guide

## Overview

Sim-bench provides three levels of dependency installation to suit different use cases.

## Installation Options

### 1. Full Install (Recommended for Most Users)

```bash
pip install -r requirements.txt
```

**Includes:**
- ✅ Core benchmarking (numpy, scipy, opencv, scikit-learn)
- ✅ Deep learning (PyTorch, torchvision)
- ✅ Visualization (matplotlib, seaborn)
- ✅ Analysis notebooks (jupyter, notebook)
- ✅ All features enabled

**Use when:**
- You want the complete sim-bench experience
- You'll analyze results in Jupyter notebooks
- You want to use ResNet50 features
- You have ~2GB disk space for dependencies

**Size**: ~2-3GB installed

---

### 2. Minimal Install (Lightweight)

```bash
pip install -r requirements-minimal.txt
```

**Includes:**
- ✅ Core benchmarking only
- ✅ HSV histograms, SIFT BoVW
- ❌ No deep learning (no PyTorch)
- ❌ No visualization (no matplotlib)
- ❌ No Jupyter notebooks

**Use when:**
- Running on servers/CI without visualization needs
- Don't need ResNet50 features
- Want minimal footprint
- Testing core functionality

**Size**: ~500MB installed

**Available methods**: `chi_square`, `emd`, `sift_bovw` (no `resnet50`)

---

### 3. Development Install (Everything)

```bash
pip install -r requirements-dev.txt
```

**Includes:**
- ✅ Everything from requirements.txt
- ✅ Testing tools (pytest, pytest-cov)
- ✅ Code quality (black, flake8, mypy)
- ✅ Documentation (sphinx)
- ✅ JupyterLab (better than notebook)
- ✅ Interactive widgets

**Use when:**
- Contributing to sim-bench
- Running tests
- Building documentation
- Using JupyterLab

**Size**: ~3-4GB installed

---

## Dependency Breakdown

### Core (Always Required)

| Package | Purpose | Size |
|---------|---------|------|
| `numpy` | Array operations, features | ~50MB |
| `scipy` | Distance measures (Wasserstein) | ~80MB |
| `scikit-learn` | SIFT clustering | ~30MB |
| `opencv-contrib-python` | Image processing, SIFT | ~200MB |
| `Pillow` | Image loading | ~10MB |
| `tqdm` | Progress bars | ~1MB |
| `PyYAML` | Configuration | ~1MB |
| `pandas` | Result storage | ~50MB |

**Total**: ~420MB

### Optional: Deep Learning

| Package | Purpose | Size |
|---------|---------|------|
| `torch` | PyTorch (ResNet50) | ~800MB |
| `torchvision` | Pre-trained models | ~300MB |

**Total**: ~1.1GB (large!)

### Optional: Visualization

| Package | Purpose | Size |
|---------|---------|------|
| `matplotlib` | Plotting | ~50MB |
| `seaborn` | Statistical plots | ~5MB |

**Total**: ~55MB

### Optional: Jupyter

| Package | Purpose | Size |
|---------|---------|------|
| `jupyter` | Notebook interface | ~10MB |
| `notebook` | Classic notebook | ~20MB |

**Total**: ~30MB

### Dev-Only

| Package | Purpose | Size |
|---------|---------|------|
| `pytest` | Testing framework | ~5MB |
| `black` | Code formatter | ~3MB |
| `jupyterlab` | Modern notebook UI | ~50MB |

**Total**: ~100MB

---

## Same Environment vs. Separate

### Recommendation: **Same Environment** ✅

**Why:**
- The notebook analyzes results from sim-bench runs
- No conflicts between packages
- Jupyter/seaborn are lightweight (~85MB total)
- Convenient workflow

```bash
# Single environment workflow
pip install -r requirements.txt
python -m sim_bench.cli --quick --methods chi_square
jupyter notebook analyze_results.ipynb  # Works immediately
```

### When to Use Separate Environment

**Only if:**
1. **Disk space is critical** (< 3GB available)
2. **Running on server without GUI** (no visualization needed)
3. **CI/automated testing** (minimal deps only)

```bash
# Environment 1: Benchmarking (minimal)
python -m venv .venv-bench
.venv-bench/bin/activate
pip install -r requirements-minimal.txt
python -m sim_bench.cli ...

# Environment 2: Analysis (full)
python -m venv .venv-analysis
.venv-analysis/bin/activate
pip install -r requirements.txt
jupyter notebook analyze_results.ipynb
```

**Drawback**: Managing two environments is cumbersome.

---

## Platform-Specific Notes

### Windows

All packages work on Windows. PyTorch auto-detects CUDA if available.

```bash
# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### macOS (Apple Silicon)

PyTorch has special ARM builds:
```bash
pip install -r requirements.txt  # Automatically uses ARM build
```

### Linux

Standard installation works. For GPU:
```bash
# CUDA 11.8 (check your CUDA version)
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Upgrading Dependencies

### Full Upgrade

```bash
pip install --upgrade -r requirements.txt
```

### Selective Upgrade

```bash
# Upgrade specific package
pip install --upgrade numpy

# Upgrade visualization only
pip install --upgrade matplotlib seaborn
```

### Check Versions

```bash
pip list | grep -E "numpy|torch|jupyter"
```

---

## Troubleshooting

### "No module named 'torch'"

You're using minimal install. Either:
```bash
# Add PyTorch
pip install torch>=2.0 torchvision>=0.15

# Or use full install
pip install -r requirements.txt
```

### "No module named 'seaborn'"

Notebook visualization needs seaborn:
```bash
pip install seaborn>=0.13
```

### "Command 'jupyter' not found"

Install Jupyter:
```bash
pip install jupyter notebook
```

### PyTorch version conflicts

Uninstall and reinstall:
```bash
pip uninstall torch torchvision
pip install -r requirements.txt
```

---

## Summary

| Use Case | Requirements File | Size | Includes Notebook? |
|----------|------------------|------|-------------------|
| **General use** | `requirements.txt` | ~3GB | ✅ Yes |
| **Server/CI** | `requirements-minimal.txt` | ~500MB | ❌ No |
| **Development** | `requirements-dev.txt` | ~4GB | ✅ Yes + more |

**Recommendation**: Use `requirements.txt` for the same environment - it's simpler and the notebook packages are lightweight relative to the total install size.

