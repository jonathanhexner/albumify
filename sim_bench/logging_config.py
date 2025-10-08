"""
Logging configuration for sim-bench.
Provides both console and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "sim_bench",
    log_file: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True
) -> logging.Logger:
    """
    Set up logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (None = auto-generate in output dir)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_experiment_start(logger: logging.Logger, config: dict) -> None:
    """Log experiment start with configuration."""
    logger.info("=" * 80)
    logger.info("EXPERIMENT START")
    logger.info("=" * 80)
    logger.info(f"Dataset: {config.get('dataset', 'N/A')}")
    logger.info(f"Methods: {config.get('methods', [])}")
    logger.info(f"Metrics: {config.get('metrics', [])}")
    logger.info(f"Sampling: {config.get('sampling', {})}")
    logger.info(f"Caching: {config.get('cache_features', True)}")
    logger.info("=" * 80)


def log_method_start(logger: logging.Logger, method_name: str, method_config: dict) -> None:
    """Log method execution start."""
    logger.info("-" * 80)
    logger.info(f"METHOD: {method_name}")
    logger.info(f"Config: {method_config}")
    logger.info("-" * 80)


def log_results(logger: logging.Logger, method_name: str, metrics: dict) -> None:
    """Log method results."""
    logger.info(f"RESULTS for {method_name}:")
    for metric_name, metric_value in metrics.items():
        if metric_name not in ['num_queries', 'num_images']:
            logger.info(f"  {metric_name:20s}: {metric_value:.6f}")
        else:
            logger.info(f"  {metric_name:20s}: {metric_value}")


def log_experiment_end(logger: logging.Logger) -> None:
    """Log experiment end."""
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)

