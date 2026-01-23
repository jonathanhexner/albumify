"""
Dataset classes for face recognition training.

Supports:
- MXNet RecordIO format (CASIA-WebFace, MS1M, etc.)
- Standard folder format (identity_id/image.jpg)
"""
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class MXNetRecordDataset(Dataset):
    """
    Dataset for MXNet RecordIO format (.rec, .idx files).

    This format is commonly used for large-scale face datasets like
    CASIA-WebFace, MS1M, etc.

    The .rec file contains image data in binary format with embedded labels.
    The .idx file contains byte offsets for random access.

    Note: Labels are extracted directly from the rec file header (bytes 4-8),
    NOT from the .lst file (which may have mismatches).
    """

    def __init__(self, rec_path: str, transform=None, num_classes: int = None):
        """
        Initialize dataset from RecordIO files.

        Args:
            rec_path: Path to .rec file (will also load .idx file)
            transform: Optional torchvision transform
            num_classes: Expected number of classes (for validation)
        """
        self.rec_path = Path(rec_path)
        self.idx_path = self.rec_path.with_suffix('.idx')
        self.transform = transform

        if not self.rec_path.exists():
            raise FileNotFoundError(f"RecordIO file not found: {self.rec_path}")
        if not self.idx_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")

        # Load index file (byte offsets)
        all_offsets = self._load_idx()
        logger.info(f"Loaded {len(all_offsets)} index entries from {self.idx_path}")

        # Open record file
        self.rec_file = open(self.rec_path, 'rb')

        # Pre-scan to find valid image records (flag=0)
        # and extract labels from rec file headers
        self.samples = self._build_valid_samples(all_offsets)
        logger.info(f"Found {len(self.samples)} valid image records")

        # Validate num_classes
        self.num_classes = num_classes
        actual_labels = [s[1] for s in self.samples]
        actual_classes = len(set(actual_labels))
        logger.info(f"Dataset has {actual_classes} unique classes (labels {min(actual_labels)}-{max(actual_labels)})")
        if num_classes and actual_classes != num_classes:
            logger.warning(f"Expected {num_classes} classes, found {actual_classes}")

    def _load_idx(self) -> List[int]:
        """Load byte offsets from index file."""
        offsets = []
        with open(self.idx_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    byte_offset = int(parts[1])
                    offsets.append(byte_offset)
        return offsets

    def _build_valid_samples(self, all_offsets: List[int]) -> List[Tuple[int, int]]:
        """
        Scan all records and build list of (offset, label) for valid images.

        Valid image records have flag=0 in their header.
        Labels are extracted from the rec file header (bytes 4-8 as float).
        """
        samples = []

        for offset in all_offsets:
            self.rec_file.seek(offset)

            # Read record header
            magic_bytes = self.rec_file.read(4)
            if len(magic_bytes) < 4:
                continue

            magic = struct.unpack('I', magic_bytes)[0]
            if magic != 0xced7230a:
                continue

            length_flag = struct.unpack('I', self.rec_file.read(4))[0]
            length = length_flag & ((1 << 29) - 1)

            # Read the data header: flag(4) + label(4)
            if length < 8:
                continue

            header_data = self.rec_file.read(8)
            data_flag = struct.unpack('I', header_data[0:4])[0]
            label = int(struct.unpack('f', header_data[4:8])[0])

            # Flag 0 = image record, other flags = metadata
            if data_flag == 0:
                samples.append((offset, label))

        return samples

    def _read_record(self, offset: int) -> bytes:
        """
        Read a single record from the RecordIO file.

        Returns:
            Image bytes (JPEG/PNG encoded)
        """
        self.rec_file.seek(offset)

        # Read record header
        magic = struct.unpack('I', self.rec_file.read(4))[0]
        if magic != 0xced7230a:
            raise ValueError(f"Invalid magic number at offset {offset}: {hex(magic)}")

        length_flag = struct.unpack('I', self.rec_file.read(4))[0]
        cflag = (length_flag >> 29) & 7
        length = length_flag & ((1 << 29) - 1)

        # Read data (may span multiple records for large images)
        if cflag == 0:
            data = self.rec_file.read(length)
        else:
            # Multi-part record
            data = self.rec_file.read(length)
            while cflag != 0:
                magic = struct.unpack('I', self.rec_file.read(4))[0]
                length_flag = struct.unpack('I', self.rec_file.read(4))[0]
                cflag = (length_flag >> 29) & 7
                length = length_flag & ((1 << 29) - 1)
                data += self.rec_file.read(length)

        # Skip 24-byte header: flag(4) + label(4) + id(8) + id2(8)
        # Image data (JPEG/PNG) starts at offset 24
        header_size = 24
        image_data = data[header_size:]

        return image_data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        offset, label = self.samples[idx]
        image_bytes = self._read_record(offset)

        # Decode JPEG/PNG bytes to PIL Image
        import io
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'index': idx
        }

    def __del__(self):
        if hasattr(self, 'rec_file') and self.rec_file:
            self.rec_file.close()


class FolderFaceDataset(Dataset):
    """
    Dataset for standard folder structure: root/identity_id/image.jpg

    Each subfolder represents one identity class.
    """

    def __init__(self, root_dir: str, transform=None, extensions: tuple = ('.jpg', '.jpeg', '.png')):
        """
        Initialize dataset from folder structure.

        Args:
            root_dir: Root directory containing identity folders
            transform: Optional torchvision transform
            extensions: Valid image extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        # Build class mapping and sample list
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.samples = self._scan_samples()
        logger.info(f"Found {len(self.samples)} images in {self.num_classes} classes")

    def _scan_samples(self) -> List[Tuple[Path, int]]:
        """Scan all images and their labels."""
        samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    samples.append((img_path, class_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': str(img_path)
        }


def create_face_dataset(config: dict, transform=None) -> Dataset:
    """
    Factory function to create face dataset based on config.

    Args:
        config: Data configuration with keys:
            - data_format: 'recordio' or 'folder'
            - rec_path: Path to .rec file (for recordio format)
            - root_dir: Root directory (for folder format)
            - num_classes: Number of identity classes
        transform: Optional transform to apply

    Returns:
        Dataset instance
    """
    data_format = config.get('data_format', 'recordio')

    if data_format == 'recordio':
        return MXNetRecordDataset(
            rec_path=config['rec_path'],
            transform=transform,
            num_classes=config.get('num_classes')
        )
    elif data_format == 'folder':
        return FolderFaceDataset(
            root_dir=config['root_dir'],
            transform=transform
        )
    else:
        raise ValueError(f"Unknown data format: {data_format}")


def create_train_val_split(dataset: Dataset, val_ratio: float = 0.1,
                           seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Create train/val split indices.

    For face recognition, we typically keep all identities in both sets
    but split images randomly.

    Args:
        dataset: Full dataset
        val_ratio: Fraction of samples for validation
        seed: Random seed

    Returns:
        (train_indices, val_indices)
    """
    n_samples = len(dataset)
    indices = list(range(n_samples))

    np.random.seed(seed)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_ratio)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices, val_indices
