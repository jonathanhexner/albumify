# Quick Start: Kaggle Training

## 1. Create Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → **GPU T4 x2** ✅
4. Settings → Internet → **On** ✅

## 2. Add PhotoTriage Dataset

**Use the existing Kaggle dataset** (no upload needed!):

1. In the notebook, click "+ Add Data" (right sidebar)
2. Search for "**triage**" or "**ericwolter/triage**"
3. Click "Add" on the PhotoTriage dataset
4. Dataset will be available at `/kaggle/input/triage/`

**Dataset URL**: https://www.kaggle.com/datasets/ericwolter/triage

## 3. Upload Training Notebook

**Option A: Upload File**
- File → Upload Notebook
- Select `kaggle_siamese_training.ipynb`

**Option B: Copy-Paste**
- Create new cells and copy content from the notebook

## 4. Update Configuration

In Cell 5, update the GitHub URL:
```python
!git clone https://github.com/YOUR_USERNAME/sim-bench.git
```

The dataset path is already configured to use `/kaggle/input/triage/`

## 5. Run Training

### Quick Test (5 minutes)
Run cells 1-10 to test with 10% of data

### Full Training (2-3 hours)
Run all cells for complete training

## 6. Download Results

After training completes:
1. Scroll to Cell 16
2. Run it to create `siamese_training_results.zip`
3. Click the output file to download
4. Or: Click "Save Version" to preserve all outputs

## What You Get

The results zip contains:
- ✅ `best_model.pt` - Trained model weights
- ✅ `config.yaml` - Training configuration
- ✅ `results.json` - Test accuracy and loss
- ✅ `training_history.json` - Per-epoch metrics
- ✅ `training_curves.png` - Loss/accuracy plots
- ✅ `training.log` - Complete training logs

## Expected Performance

| Model | Test Accuracy | Training Time | GPU Memory |
|-------|--------------|---------------|------------|
| VGG16 | ~65-70% | 2-3 hours | ~8GB |
| ResNet50 | ~68-72% | 3-4 hours | ~6GB |

## Troubleshooting

**"Dataset not found"**
→ Add dataset to notebook (step 4)

**"Out of memory"**
→ Reduce batch_size to 16 or 8 in Cell 8

**"CUDA not available"**
→ Enable GPU in Settings → Accelerator

**"Import errors"**
→ Restart kernel and run all cells from beginning

## Tips

🔥 **Enable GPU** - Settings → Accelerator → GPU T4 x2

⚡ **Quick Test First** - Use 10% data to verify setup (Cell 10)

💾 **Save Regularly** - Click "Save Version" to preserve outputs

📊 **Monitor Progress** - Watch GPU utilization in right panel

⏰ **Be Patient** - Full training takes 2-4 hours

## Next Steps

After successful training:

1. **Compare architectures**: Try both VGG16 and ResNet50
2. **Tune hyperparameters**: Adjust learning rate, batch size
3. **Visualize results**: Check `training_curves.png`
4. **Use the model**: Load `best_model.pt` for inference

Happy training! 🚀

