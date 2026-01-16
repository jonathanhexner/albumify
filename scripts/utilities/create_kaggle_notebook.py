#!/usr/bin/env python3
"""
Generate corrected Kaggle notebook with command-line overrides.

Usage:
    python create_kaggle_notebook.py
    
Output:
    kaggle_siamese_training_corrected.ipynb
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Siamese CNN Training for PhotoTriage\n",
                "End-to-end training of VGG16/ResNet50 for pairwise image quality ranking\n",
                "\n",
                "## Setup\n",
                "This notebook:\n",
                "1. Uses the existing PhotoTriage dataset on Kaggle (https://www.kaggle.com/datasets/ericwolter/triage)\n",
                "2. Clones and installs sim_bench package\n",
                "3. **Uses YAML configs directly with command-line overrides**\n",
                "4. Trains Siamese CNN + MLP end-to-end with **differential learning rates**\n",
                "5. Saves results and plots\n",
                "\n",
                "## Before Running\n",
                "1. **Add the dataset**: Click \"+ Add Data\" → Search \"triage\" → Add the dataset by ericwolter\n",
                "2. **Enable GPU**: Settings → Accelerator → GPU T4 x2\n",
                "3. **Enable Internet**: Settings → Internet → On (to clone GitHub repo)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU availability\n",
                "import torch\n",
                "print(f\"PyTorch version: {torch.__version__}\")\n",
                "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"CUDA device: {torch.cuda.get_device_name(0)}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Download PhotoTriage Dataset\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Use existing PhotoTriage dataset from Kaggle\n",
                "# Dataset: https://www.kaggle.com/datasets/ericwolter/triage\n",
                "# Add it to your notebook: Click \"+ Add Data\" → Search \"triage\" → Add\n",
                "\n",
                "from pathlib import Path\n",
                "import os\n",
                "\n",
                "# Check for the dataset\n",
                "dataset_path = Path('/kaggle/input/triage')\n",
                "if dataset_path.exists():\n",
                "    print(f\"✓ Dataset found: {dataset_path}\")\n",
                "    print(\"\\nDataset contents:\")\n",
                "    !ls -lh {dataset_path}\n",
                "    \n",
                "    # Check subdirectories\n",
                "    for subdir in ['train_val', 'test']:\n",
                "        subdir_path = dataset_path / subdir\n",
                "        if subdir_path.exists():\n",
                "            img_count = len(list(subdir_path.rglob('*.JPG'))) + len(list(subdir_path.rglob('*.jpg')))\n",
                "            print(f\"\\n{subdir}: {img_count} images\")\n",
                "else:\n",
                "    print(\"❌ Dataset not found!\")\n",
                "    print(\"\\nTo add the dataset:\")\n",
                "    print(\"1. Click '+ Add Data' in the right sidebar\")\n",
                "    print(\"2. Search for 'triage' or 'ericwolter/triage'\")\n",
                "    print(\"3. Click 'Add' on the PhotoTriage dataset\")\n",
                "    print(\"4. Re-run this cell\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Clone and Install sim_bench\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository\n",
                "!git clone https://github.com/YOUR_USERNAME/sim-bench.git /kaggle/working/sim-bench\n",
                "%cd /kaggle/working/sim-bench\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                "%pip install -e .\n",
                "%pip install pandas pillow pyyaml matplotlib seaborn\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Verify Configuration Files\n",
                "\n",
                "We use the YAML configs directly from the repo with command-line overrides.\n",
                "No copying or modification needed - keeps configs in sync!\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import yaml\n",
                "from pathlib import Path\n",
                "\n",
                "# Check available configs\n",
                "configs_dir = Path('/kaggle/working/sim-bench/configs/siamese_e2e')\n",
                "\n",
                "print(\"Available configs from repo:\\n\")\n",
                "for config_file in sorted(configs_dir.glob('*.yaml')):\n",
                "    with open(config_file, 'r') as f:\n",
                "        config = yaml.safe_load(f)\n",
                "    \n",
                "    print(f\"✓ {config_file.name}\")\n",
                "    print(f\"  Model: {config['model']['cnn_backbone']}\")\n",
                "    print(f\"  Batch size: {config['training']['batch_size']}\")\n",
                "    print(f\"  Base LR: {config['training']['learning_rate']} (backbone)\")\n",
                "    print(f\"  Head LR: {config['training']['learning_rate'] * 10} (10x)\")\n",
                "    print(f\"  Differential LR: {config['training'].get('differential_lr', False)}\")\n",
                "    print(f\"  Max epochs: {config['training']['max_epochs']}\")\n",
                "    print()\n",
                "\n",
                "print(\"=\"*70)\n",
                "print(\"We'll override only platform-specific settings via command-line:\")\n",
                "print(\"  --data-dir /kaggle/input/triage\")\n",
                "print(\"  --device cuda\")\n",
                "print(\"  --output-dir /kaggle/working/outputs/...\")\n",
                "print(\"=\"*70)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Quick Test (Optional)\n",
                "\n",
                "Test with 10% of data to verify everything works.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run quick test with 10% of data to verify everything works\n",
                "!python -m sim_bench.training.train_siamese_e2e \\\n",
                "    --config configs/siamese_e2e/vgg16.yaml \\\n",
                "    --data-dir /kaggle/input/triage \\\n",
                "    --device cuda \\\n",
                "    --output-dir /kaggle/working/outputs/quick_test \\\n",
                "    --quick-experiment 0.1\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Full Training\n",
                "\n",
                "Choose either VGG16 or ResNet50 (or train both for comparison).\n",
                "\n",
                "**Expected Results (with differential LR):**\n",
                "- Epoch 1: ~62% train accuracy, ~70% validation accuracy\n",
                "- Epoch 10+: ~70%+ train/validation accuracy\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train VGG16 (paper replication)\n",
                "!python -m sim_bench.training.train_siamese_e2e \\\n",
                "    --config configs/siamese_e2e/vgg16.yaml \\\n",
                "    --data-dir /kaggle/input/triage \\\n",
                "    --device cuda \\\n",
                "    --output-dir /kaggle/working/outputs/siamese_e2e_vgg16\n",
                "\n",
                "# Or train ResNet50 (uncomment to use)\n",
                "# !python -m sim_bench.training.train_siamese_e2e \\\n",
                "#     --config configs/siamese_e2e/resnet50.yaml \\\n",
                "#     --data-dir /kaggle/input/triage \\\n",
                "#     --device cuda \\\n",
                "#     --output-dir /kaggle/working/outputs/siamese_e2e_resnet50\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Load and Visualize Results\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "# Load results\n",
                "output_dir = Path('/kaggle/working/outputs/siamese_e2e_vgg16')\n",
                "results_file = output_dir / 'results.json'\n",
                "history_file = output_dir / 'training_history.json'\n",
                "\n",
                "if results_file.exists():\n",
                "    with open(results_file, 'r') as f:\n",
                "        results = json.load(f)\n",
                "    \n",
                "    print(\"\\n\" + \"=\"*50)\n",
                "    print(\"FINAL RESULTS\")\n",
                "    print(\"=\"*50)\n",
                "    print(f\"Test Accuracy:  {results['test_acc']:.3f}\")\n",
                "    print(f\"Test Loss:      {results['test_loss']:.4f}\")\n",
                "    \n",
                "    # Check model checkpoint\n",
                "    model_file = output_dir / 'best_model.pt'\n",
                "    if model_file.exists():\n",
                "        checkpoint = torch.load(model_file, map_location='cpu')\n",
                "        print(f\"\\nBest model from epoch: {checkpoint['epoch'] + 1}\")\n",
                "        print(f\"Validation accuracy: {checkpoint['val_acc']:.3f}\")\n",
                "    \n",
                "    # Display training curves\n",
                "    curves_file = output_dir / 'training_curves.png'\n",
                "    if curves_file.exists():\n",
                "        print(\"\\n\" + \"=\"*50)\n",
                "        print(\"TRAINING CURVES\")\n",
                "        print(\"=\"*50)\n",
                "        from IPython.display import Image, display\n",
                "        display(Image(filename=str(curves_file)))\n",
                "    \n",
                "    # Show training history\n",
                "    if history_file.exists():\n",
                "        with open(history_file, 'r') as f:\n",
                "            history = json.load(f)\n",
                "        \n",
                "        print(\"\\n\" + \"=\"*50)\n",
                "        print(\"TRAINING HISTORY (Last 5 Epochs)\")\n",
                "        print(\"=\"*50)\n",
                "        print(\"Epoch | Train Loss | Train Acc | Val Loss | Val Acc\")\n",
                "        print(\"-\" * 50)\n",
                "        for i in range(max(0, len(history['train_loss']) - 5), len(history['train_loss'])):\n",
                "            print(f\"{i+1:5d} | {history['train_loss'][i]:10.4f} | {history['train_acc'][i]:9.3f} | \"\n",
                "                  f\"{history['val_loss'][i]:8.4f} | {history['val_acc'][i]:7.3f}\")\n",
                "else:\n",
                "    print(f\"❌ Results not found: {results_file}\")\n",
                "    print(\"\\nMake sure training completed successfully.\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Package and Download Results\n",
                "\n",
                "Creates a zip file with all training outputs for download.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "import os\n",
                "\n",
                "# Create a zip file of all results\n",
                "output_zip = '/kaggle/working/siamese_training_results'\n",
                "shutil.make_archive(output_zip, 'zip', '/kaggle/working/outputs')\n",
                "\n",
                "zip_file = output_zip + '.zip'\n",
                "print(f\"✓ Results packaged: {zip_file}\")\n",
                "print(f\"  File size: {os.path.getsize(zip_file) / 1024 / 1024:.2f} MB\")\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"Download this file from the Kaggle output section\")\n",
                "print(\"=\"*60)\n",
                "print(\"\\nContents include:\")\n",
                "print(\"  - best_model.pt (trained model weights)\")\n",
                "print(\"  - config.yaml (training configuration)\")\n",
                "print(\"  - results.json (final test results)\")\n",
                "print(\"  - training_history.json (per-epoch metrics)\")\n",
                "print(\"  - training_curves.png (loss/accuracy plots)\")\n",
                "print(\"  - training.log (complete training logs)\")\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

if __name__ == "__main__":
    output_file = "kaggle_siamese_training_corrected.ipynb"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Created: {output_file}")
    print("\nThis notebook:")
    print("  - Uses YAML configs directly from repo")
    print("  - Uses command-line overrides for Kaggle-specific settings")
    print("  - Has correct differential learning rates")
    print("  - Includes enhanced results visualization")
    print("\nUpload this notebook to Kaggle to use it!")

