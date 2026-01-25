"""Plotting utilities for face recognition training."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_curves(history: dict, output_dir: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Classification accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Verification accuracy plot (if available)
    if 'lfw_acc' in history and any(v is not None for v in history['lfw_acc']):
        val_epochs = [i + 1 for i, v in enumerate(history['lfw_acc']) if v is not None]
        val_accs = [v for v in history['lfw_acc'] if v is not None]
        axes[1, 0].plot(val_epochs, val_accs, 'g-o', label='LFW Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('LFW Verification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 100])
    else:
        axes[1, 0].text(0.5, 0.5, 'No verification data', ha='center', va='center')
        axes[1, 0].set_title('LFW Verification Accuracy')

    # Learning rate plot
    axes[1, 1].plot(epochs, history['learning_rate'], 'k-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    logger.info(f"Training curves saved to {output_dir / 'training_curves.png'}")
    plt.close()


def plot_verification_roc(fpr, tpr, auc_score, output_path: Path):
    """Plot ROC curve for verification."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Verification ROC Curve')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
