"""Inspect external MyDataset to understand train/val splitting logic."""
import sys
import inspect
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')
from data.dataloader import MyDataset

image_root = r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs'
train_ds = MyDataset(train=True, image_root=image_root, seed=42)
val_ds = MyDataset(train=False, image_root=image_root, seed=42)

logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
logger.info(f"Split ratio: {len(train_ds)/(len(train_ds)+len(val_ds)):.1%} train")
logger.info(f"\nSource:\n{inspect.getsource(MyDataset)}")
