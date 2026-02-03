# Running sim-bench on Kaggle

This guide explains how to run Siamese CNN training on Kaggle Notebooks with free GPU access.

## Prerequisites

### 1. PhotoTriage Dataset (Already on Kaggle!)

**Great news!** The PhotoTriage dataset is already available on Kaggle:

**📦 Dataset**: https://www.kaggle.com/datasets/ericwolter/triage

**To use it:**
1. Open your Kaggle notebook
2. Click "+ Add Data" in the right sidebar
3. Search for "**triage**" or "**ericwolter/triage**"
4. Click "Add"

The dataset will be available at `/kaggle/input/triage/`

**No upload needed!** ✨

**Note**: The `photo_triage_pairs_embedding_labels.csv` file is included in the sim-bench package (`data/phototriage/`), so it's automatically available.

### 2. GitHub Repository
Make sure your sim-bench repository is public or you have access to clone it.

## Usage

### Quick Start

1. **Create a new Kaggle Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Enable GPU: Settings → Accelerator → GPU T4 x2

2. **Upload the notebook**
   - Upload `kaggle_siamese_training.ipynb` from this repository
   - Or copy-paste the cells

3. **Add the PhotoTriage dataset**
   - In the notebook, click "+ Add Data"
   - Search for "triage" or "ericwolter/triage"
   - Click "Add" on the PhotoTriage dataset

4. **Update the notebook**
   - Cell 5: Update the GitHub URL to your repository
   - Dataset path is already configured for `/kaggle/input/triage/`

5. **Run all cells**
   - The notebook will:
     - Check GPU availability
     - Download and verify the dataset
     - Clone and install sim-bench
     - Run a quick test (10% of data)
     - Run full training
     - Generate and save plots
     - Package results for download

### Configuration

Edit the config in Cell 8 to customize training:

```python
vgg16_config = {
    'model': {
        'cnn_backbone': 'vgg16',  # or 'resnet50'
        'mlp_hidden_dims': [128, 128],
        'dropout': 0.0,
        'activation': 'tanh'
    },
    'training': {
        'batch_size': 32,  # Adjust based on GPU memory
        'learning_rate': 0.001,
        'max_epochs': 30
    }
}
```

### Outputs

The notebook saves:
- `best_model.pt` - Best model checkpoint
- `config.yaml` - Training configuration
- `results.json` - Final test results
- `training_history.json` - Loss/accuracy per epoch
- `training_curves.png` - Training/validation plots
- `training.log` - Complete training logs (all epochs + batches)
- `siamese_training_results.zip` - All outputs packaged

### Tips

1. **GPU Memory**
   - Start with batch_size=16 or 32
   - VGG16 uses more memory than ResNet50
   - Monitor GPU usage in Kaggle's right panel

2. **Quick Testing**
   - Use `quick_experiment: 0.1` to test with 10% of data
   - Reduces training time from hours to minutes
   - Good for debugging before full training

3. **Saving Results**
   - Kaggle auto-saves outputs in `/kaggle/working/`
   - Download the zip file before closing the notebook
   - Or use Kaggle's "Save Version" to preserve outputs

4. **Training Time**
   - Quick test (10% data): ~5-10 minutes
   - Full training VGG16: ~2-3 hours (30 epochs)
   - Full training ResNet50: ~3-4 hours (30 epochs)

### Troubleshooting

**Dataset not found**
- Click "+ Add Data" and search for "triage"
- Add the dataset by ericwolter: https://www.kaggle.com/datasets/ericwolter/triage
- Dataset should appear in `/kaggle/input/triage/`

**Out of memory**
- Reduce batch_size (try 16 or 8)
- Use ResNet50 instead of VGG16 (smaller memory footprint)
- Enable gradient checkpointing (advanced)

**Training too slow**
- Ensure GPU is enabled (Settings → Accelerator)
- Check GPU utilization (should be >80%)
- Reduce log_interval to reduce logging overhead

**Import errors**
- Ensure all dependencies are installed in Cell 6
- Check that sim-bench installation succeeded
- Restart kernel and run all cells again

## Expected Results

### VGG16 (Paper Architecture)
- Test Accuracy: ~65-70%
- Training time: ~2-3 hours (GPU)
- Memory usage: ~8GB GPU

### ResNet50
- Test Accuracy: ~68-72%
- Training time: ~3-4 hours (GPU)
- Memory usage: ~6GB GPU

## Next Steps

After training completes:
1. Download `siamese_training_results.zip`
2. Extract and review:
   - `training_curves.png` - Visual inspection of convergence
   - `results.json` - Final test accuracy
   - `best_model.pt` - Use for inference
3. Compare different architectures (VGG16 vs ResNet50)
4. Try different hyperparameters

## Resources

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [PhotoTriage Paper](https://arxiv.org/abs/2103.00430)
- [sim-bench Repository](https://github.com/YOUR_USERNAME/sim-bench)

