"""
Binary classifier for PhotoTriage pairwise preferences.

Learns to predict which image users prefer from CLIP embeddings.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BinaryClassifierConfig:
    """Configuration for PhotoTriage binary classifier training."""

    # Model architecture
    clip_model: str = "ViT-B-32"
    clip_checkpoint: str = "laion2b_s34b_b79k"
    freeze_backbone: bool = True
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.3

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
    precompute_embeddings: bool = True

    # Output
    output_dir: str = "outputs/trained_models/phototriage_binary"
    save_plots: bool = True
    save_best_only: bool = True

    # Logging
    log_interval: int = 10  # Log every N batches

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BinaryClassifierConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BinaryClassifierConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)


class PhotoTriageBinaryClassifier(nn.Module):
    """
    Binary classifier for predicting which image wins in a pair.

    Architecture:
        Image Pair → CLIP Encoder (frozen) → [emb1; emb2] → MLP → logits (2 classes)

    Args:
        config: BinaryClassifierConfig with model settings
    """

    def __init__(self, config: BinaryClassifierConfig):
        super().__init__()

        self.config = config

        # Load CLIP model
        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip is required. Install: pip install open-clip-torch")

        logger.info(f"Loading CLIP: {config.clip_model} ({config.clip_checkpoint})")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.clip_model,
            pretrained=config.clip_checkpoint
        )

        # Get embedding dimension
        if config.clip_model.startswith("ViT-B"):
            self.embed_dim = 512
        elif config.clip_model.startswith("ViT-L"):
            self.embed_dim = 768
        elif config.clip_model.startswith("ViT-H"):
            self.embed_dim = 1024
        else:
            # Try to infer from model
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                dummy_out = self.clip_model.encode_image(dummy_input)
            self.embed_dim = dummy_out.shape[-1]

        logger.info(f"CLIP embedding dimension: {self.embed_dim}")

        # Freeze CLIP if requested
        if config.freeze_backbone:
            logger.info("Freezing CLIP backbone")
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        else:
            logger.info("CLIP backbone will be fine-tuned")

        # Build MLP classifier
        # Input: concatenated embeddings [emb1; emb2] = 2 * embed_dim
        # If mlp_hidden_dims is empty [], this becomes a simple linear classifier
        mlp_layers = []
        in_dim = 2 * self.embed_dim

        for hidden_dim in config.mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim

        # Output layer: 2 classes (image1 wins, image2 wins)
        mlp_layers.append(nn.Linear(in_dim, 2))

        self.classifier = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(config.device)

        # Log model info
        self._log_model_info()

    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _log_model_info(self):
        """Log model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info("PhotoTriage Binary Classifier")
        logger.info("=" * 60)
        logger.info(f"CLIP Model: {self.config.clip_model}")
        logger.info(f"Embedding Dim: {self.embed_dim}")
        logger.info(f"MLP Hidden Dims: {self.config.mlp_hidden_dims}")
        logger.info(f"Dropout: {self.config.dropout}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Frozen Parameters: {total_params - trainable_params:,}")
        logger.info("=" * 60)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to CLIP embedding.

        Args:
            image: Image tensor (batch_size, 3, 224, 224)

        Returns:
            Embedding (batch_size, embed_dim), L2 normalized
        """
        if self.config.freeze_backbone:
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image)
        else:
            embedding = self.clip_model.encode_image(image)

        # L2 normalize (CLIP is trained with normalized embeddings)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def forward(
        self,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        emb1: Optional[torch.Tensor] = None,
        emb2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image1: First image (batch_size, 3, 224, 224) OR None if emb1 provided
            image2: Second image (batch_size, 3, 224, 224) OR None if emb2 provided
            emb1: Pre-computed embedding for image1 (batch_size, embed_dim) OR None
            emb2: Pre-computed embedding for image2 (batch_size, embed_dim) OR None

        Returns:
            Logits (batch_size, 2) - [logit_image1_wins, logit_image2_wins]
        """
        # Get embeddings
        if emb1 is None:
            if image1 is None:
                raise ValueError("Must provide either image1 or emb1")
            emb1 = self.encode_image(image1)

        if emb2 is None:
            if image2 is None:
                raise ValueError("Must provide either image2 or emb2")
            emb2 = self.encode_image(image2)

        # Concatenate embeddings
        combined = torch.cat([emb1, emb2], dim=-1)  # (batch_size, 2*embed_dim)

        # Classify
        logits = self.classifier(combined)  # (batch_size, 2)

        return logits

    def predict(
        self,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        emb1: Optional[torch.Tensor] = None,
        emb2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict winner and probabilities.

        Args:
            Same as forward()

        Returns:
            predictions: Predicted class (batch_size,) - 0=image1 wins, 1=image2 wins
            probabilities: Class probabilities (batch_size, 2)
        """
        logits = self.forward(image1, image2, emb1, emb2)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'embed_dim': self.embed_dim
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'PhotoTriageBinaryClassifier':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        config = BinaryClassifierConfig.from_dict(checkpoint['config'])

        if device is not None:
            config.device = device

        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {path}")

        return model


class BinaryClassifierTrainer:
    """
    Trainer for PhotoTriage binary classifier.

    Handles training loop, validation, early stopping, and visualization.
    """

    def __init__(
        self,
        model: PhotoTriageBinaryClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: BinaryClassifierConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

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

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            emb1 = batch['emb1'].to(self.config.device)
            emb2 = batch['emb2'].to(self.config.device)
            labels = batch['label'].to(self.config.device)

            # Forward pass
            logits = self.model(emb1=emb1, emb2=emb2)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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
                emb1 = batch['emb1'].to(self.config.device)
                emb2 = batch['emb2'].to(self.config.device)
                labels = batch['label'].to(self.config.device)

                # Forward pass
                logits = self.model(emb1=emb1, emb2=emb2)
                loss = self.criterion(logits, labels)

                # Metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

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
