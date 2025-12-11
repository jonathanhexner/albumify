"""
Series-aware classifier for PhotoTriage.

Trains on full series instead of pairs, using series-softmax loss.
This matches the actual task: selecting the best image from a series.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SeriesClassifierConfig:
    """Configuration for PhotoTriage series classifier training."""

    # Model architecture
    clip_model: str = "ViT-B-32"
    clip_checkpoint: str = "laion2b_s34b_b79k"
    freeze_backbone: bool = True
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3
    use_transformer: bool = False  # If True, use series Transformer instead of independent scoring

    # Transformer settings (only used if use_transformer=True)
    num_transformer_layers: int = 2
    num_attention_heads: int = 8
    transformer_hidden_dim: int = 1024

    # Data paths
    csv_path: str = r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv"
    image_dir: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs"

    # Data filtering
    min_agreement: float = 0.7
    min_reviewers: int = 2

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 30
    early_stop_patience: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Caching
    embedding_cache_path: str = "outputs/cache/clip_embeddings.pkl"
    use_cache: bool = True

    # Output
    output_dir: str = "outputs/trained_models/phototriage_series"
    save_plots: bool = True
    save_best_only: bool = True

    # Logging
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SeriesClassifierConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


class PhotoTriageSeriesDataset(Dataset):
    """
    Dataset that returns full series instead of pairs.

    Each item is a series with all images and the index of the best image.
    """

    def __init__(self, series_data: List[Dict], embeddings: Dict[str, torch.Tensor]):
        """
        Args:
            series_data: List of dicts with keys:
                - 'series_id': str
                - 'images': List[str] (filenames)
                - 'best_idx': int (index of best image, 0-indexed)
            embeddings: Dict mapping filename -> embedding tensor
        """
        self.series = series_data
        self.embeddings = embeddings

        # Verify all images have embeddings
        missing = set()
        for series in self.series:
            for img in series['images']:
                if img not in embeddings:
                    missing.add(img)

        if missing:
            logger.warning(f"Missing embeddings for {len(missing)} images")
            logger.warning(f"Examples: {list(missing)[:5]}")

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        series = self.series[idx]

        # Get embeddings for all images in series
        embs = [self.embeddings[img] for img in series['images']]
        embs = torch.stack(embs)  # (num_images, embed_dim)

        return {
            'embeddings': embs,  # (num_images, embed_dim)
            'best_idx': torch.tensor(series['best_idx'], dtype=torch.long),
            'num_images': len(series['images']),
            'series_id': series['series_id']
        }


def series_collate_fn(batch):
    """
    Collate function for variable-length series.
    Pads to max length in batch.

    Args:
        batch: List of items from PhotoTriageSeriesDataset

    Returns:
        Dict with batched tensors
    """
    max_len = max(item['num_images'] for item in batch)
    embed_dim = batch[0]['embeddings'].shape[1]

    batch_embs = []
    batch_labels = []
    batch_masks = []
    batch_series_ids = []

    for item in batch:
        embs = item['embeddings']  # (num_images, embed_dim)
        num_images = item['num_images']

        # Pad to max_len
        if num_images < max_len:
            padding = torch.zeros(max_len - num_images, embed_dim)
            embs = torch.cat([embs, padding], dim=0)

        # Create mask (1 for real images, 0 for padding)
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:num_images] = 1

        batch_embs.append(embs)
        batch_labels.append(item['best_idx'])
        batch_masks.append(mask)
        batch_series_ids.append(item['series_id'])

    return {
        'embeddings': torch.stack(batch_embs),  # (batch, max_len, embed_dim)
        'best_idx': torch.stack(batch_labels),  # (batch,)
        'masks': torch.stack(batch_masks),  # (batch, max_len)
        'series_ids': batch_series_ids  # List[str]
    }


class PhotoTriageSeriesClassifier(nn.Module):
    """
    Series-aware classifier that scores all images in a series.

    Two modes:
    1. Independent scoring: Each image scored independently, then softmax
    2. Transformer: Images attend to each other before scoring
    """

    def __init__(self, config: SeriesClassifierConfig):
        super().__init__()

        self.config = config
        self.embed_dim = 512  # CLIP ViT-B/32

        if config.use_transformer:
            self._build_transformer_scorer()
        else:
            self._build_independent_scorer()

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(config.device)

        # Log model info
        self._log_model_info()

    def _build_independent_scorer(self):
        """Build MLP that scores each image independently."""
        mlp_layers = []
        in_dim = self.embed_dim

        for hidden_dim in self.config.mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            ])
            in_dim = hidden_dim

        # Output: scalar score per image
        mlp_layers.append(nn.Linear(in_dim, 1))

        self.scorer = nn.Sequential(*mlp_layers)
        self.use_transformer = False

    def _build_transformer_scorer(self):
        """Build Transformer that lets images compare to each other."""
        # Positional encoding (max 20 images per series)
        self.pos_encoding = nn.Parameter(torch.randn(1, 20, self.embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.transformer_hidden_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_transformer_layers
        )

        # Final scorer
        self.scorer = nn.Linear(self.embed_dim, 1)
        self.use_transformer = True

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info("PhotoTriage Series Classifier")
        logger.info("=" * 60)
        logger.info(f"Architecture: {'Transformer' if self.use_transformer else 'Independent MLP'}")
        logger.info(f"Embedding Dim: {self.embed_dim}")
        if not self.use_transformer:
            logger.info(f"MLP Hidden Dims: {self.config.mlp_hidden_dims}")
        else:
            logger.info(f"Transformer Layers: {self.config.num_transformer_layers}")
            logger.info(f"Attention Heads: {self.config.num_attention_heads}")
        logger.info(f"Dropout: {self.config.dropout}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info("=" * 60)

    def forward(self, embeddings: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: (batch, num_images, embed_dim)
            masks: (batch, num_images) - True for real images, False for padding

        Returns:
            scores: (batch, num_images) - raw scores (before softmax)
        """
        batch_size, num_images, embed_dim = embeddings.shape

        if self.use_transformer:
            # Add positional encoding
            pos_enc = self.pos_encoding[:, :num_images, :]
            x = embeddings + pos_enc

            # Create attention mask for transformer
            # Transformer expects True = ignore
            if masks is not None:
                attn_mask = ~masks  # Invert: True = ignore in transformer
            else:
                attn_mask = None

            # Apply transformer
            x = self.transformer(x, src_key_padding_mask=attn_mask)

            # Score each image
            scores = self.scorer(x).squeeze(-1)  # (batch, num_images)
        else:
            # Independent scoring
            # Flatten to (batch*num_images, embed_dim)
            embs_flat = embeddings.view(-1, embed_dim)

            # Score each image
            scores_flat = self.scorer(embs_flat)  # (batch*num_images, 1)
            scores = scores_flat.view(batch_size, num_images)  # (batch, num_images)

        # Mask out padding (set to -inf so softmax = 0)
        if masks is not None:
            scores = scores.masked_fill(~masks, float('-inf'))

        return scores

    def predict(self, embeddings: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict best image and probabilities.

        Args:
            embeddings: (batch, num_images, embed_dim)
            masks: (batch, num_images) - optional mask for padding

        Returns:
            predictions: Predicted best index (batch,)
            probabilities: Softmax probabilities (batch, num_images)
        """
        scores = self.forward(embeddings, masks)
        probabilities = F.softmax(scores, dim=1)  # (batch, num_images)
        predictions = torch.argmax(probabilities, dim=1)  # (batch,)

        return predictions, probabilities

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': asdict(self.config),
            'embed_dim': self.embed_dim
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'PhotoTriageSeriesClassifier':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = SeriesClassifierConfig(**checkpoint['config'])

        if device is not None:
            config.device = device

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {path}")

        return model


class SeriesClassifierTrainer:
    """
    Trainer for PhotoTriage series classifier.
    """

    def __init__(
        self,
        model: PhotoTriageSeriesClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SeriesClassifierConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Metrics tracking
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.to_yaml(self.output_dir / "config.yaml")

        logger.info("Trainer initialized")

    def series_softmax_loss(self, scores: torch.Tensor, best_idx: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss over series.

        Args:
            scores: (batch, num_images) - raw scores
            best_idx: (batch,) - index of best image in each series

        Returns:
            loss: scalar
        """
        # Apply softmax within each series
        log_probs = F.log_softmax(scores, dim=1)  # (batch, num_images)

        # Cross-entropy: -log P(best_idx)
        loss = F.nll_loss(log_probs, best_idx)

        return loss

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            embeddings = batch['embeddings'].to(self.config.device)
            best_idx = batch['best_idx'].to(self.config.device)
            masks = batch['masks'].to(self.config.device)

            # Forward pass
            scores = self.model(embeddings, masks)
            loss = self.series_softmax_loss(scores, best_idx)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(scores, dim=1)
            correct += (preds == best_idx).sum().item()
            total += best_idx.size(0)

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100*correct/total:.2f}%"
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                embeddings = batch['embeddings'].to(self.config.device)
                best_idx = batch['best_idx'].to(self.config.device)
                masks = batch['masks'].to(self.config.device)

                # Forward pass
                scores = self.model(embeddings, masks)
                loss = self.series_softmax_loss(scores, best_idx)

                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(scores, dim=1)
                correct += (preds == best_idx).sum().item()
                total += best_idx.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

        for epoch in range(1, self.config.max_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.config.max_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            logger.info(
                f"\nEpoch {epoch} Summary: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%"
            )

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                if self.config.save_best_only:
                    save_path = self.output_dir / "best_model.pt"
                    self.model.save(save_path)
                    logger.info(f"✓ New best model saved (val_acc: {100*val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epoch(s) "
                    f"(best: {100*self.best_val_acc:.2f}% at epoch {self.best_epoch})"
                )

            # Early stopping
            if self.patience_counter >= self.config.early_stop_patience:
                logger.info(f"\nEarly stopping at epoch {epoch}")
                break

        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {100*self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        logger.info(f"{'='*60}")

        # Save final model
        final_path = self.output_dir / "final_model.pt"
        self.model.save(final_path)

        # Plot training curves
        if self.config.save_plots:
            self.plot_training_curves()

    def plot_training_curves(self):
        """Plot and save training curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(epochs, [100*x for x in self.train_accs], 'b-', label='Train Acc')
        ax2.plot(epochs, [100*x for x in self.val_accs], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Training curves saved to {plot_path}")

        plt.close()
