import sys
import random
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from pathlib import Path
import colorlog


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')    # for Apple Macbook GPUs
    else:
        device = torch.device('cpu')

    device = torch.device('cpu')

    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)
    return device


def setup_logging(
    verbose: bool = True,
    log_dir: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Sets up advanced logging configuration with colored console output and rotating file logs.
    
    Args:
        verbose: If True, sets logging level to INFO, otherwise WARNING
        log_dir: Directory to store log files. If None, uses current directory
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir) if log_dir else Path.cwd() / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Color format for console output
    color_format = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console Handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            f'%(log_color)s{log_format}%(reset)s',
            datefmt=date_format,
            log_colors=color_format
        )
    )
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=log_dir / 'training.log',
        maxBytes=max_file_size,
        backupCount=backup_count,
        mode='a'
    )
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format)
    )
    root_logger.addHandler(file_handler)
    
    # Suppress verbose logging from other libraries
    for lib in ['PIL', 'gym', 'wandb', 'urllib3', 'matplotlib']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info('Logging system initialized')
    logger.info(f'Log files will be saved in: {log_dir.absolute()}')

    return logger