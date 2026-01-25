"""Utility functions for face recognition training."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logger


def create_optimizer(model, config: dict):
    """Create optimizer with optional differential learning rates."""
    opt_name = config['training']['optimizer'].lower()
    base_lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    use_diff_lr = config['training'].get('differential_lr', True)

    if use_diff_lr and hasattr(model, 'get_1x_lr_params'):
        param_groups = [
            {'params': list(model.get_1x_lr_params()), 'lr': base_lr},
            {'params': list(model.get_10x_lr_params()), 'lr': base_lr * 10}
        ]
        logger.info(f"Using differential LR: backbone={base_lr}, head={base_lr * 10}")
    else:
        param_groups = model.parameters()
        logger.info(f"Using single LR: {base_lr}")

    if opt_name == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=wd)
    else:
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)


def create_scheduler(optimizer, config: dict, num_batches_per_epoch: int = None):
    """Create learning rate scheduler with optional warmup."""
    scheduler_cfg = config['training'].get('scheduler')
    if scheduler_cfg is None:
        return None, None

    scheduler_type = scheduler_cfg.get('type', 'step')
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)

    if scheduler_type == 'step':
        step_size = scheduler_cfg.get('step_size', 10)
        gamma = scheduler_cfg.get('gamma', 0.1)
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'multistep':
        milestones = scheduler_cfg.get('milestones', [10, 20, 30])
        gamma = scheduler_cfg.get('gamma', 0.1)
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    elif scheduler_type == 'cosine':
        T_max = scheduler_cfg.get('T_max', config['training']['max_epochs'])
        eta_min = scheduler_cfg.get('eta_min', 0)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None, None

    warmup_scheduler = None
    if warmup_epochs > 0 and num_batches_per_epoch is not None:
        warmup_steps = warmup_epochs * num_batches_per_epoch
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        logger.info(f"Using {warmup_epochs} epoch warmup ({warmup_steps} steps)")

    return main_scheduler, warmup_scheduler
