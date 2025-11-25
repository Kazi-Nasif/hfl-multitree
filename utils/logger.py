"""
Logging utility for HFL MultiTree project
"""
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch.utils.tensorboard as tb


class Logger:
    """Logger for experiments"""
    
    def __init__(self, 
                 exp_name: str,
                 log_dir: str = None,
                 use_tensorboard: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            exp_name: Experiment name
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            log_level: Logging level
        """
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up log directory
        if log_dir is None:
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "results" / "logs"
        
        self.log_dir = Path(log_dir) / f"{exp_name}_{self.timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        self._setup_file_logger(log_level)
        
        # Set up TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.writer = tb.SummaryWriter(str(tb_dir))
        else:
            self.writer = None
        
        self.logger.info(f"Logger initialized for experiment: {exp_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def _setup_file_logger(self, log_level: str):
        """Set up file logging"""
        self.logger = logging.getLogger(self.exp_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = self.log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def log_metrics(self, metrics: dict, step: int):
        """
        Log metrics to both file and TensorBoard
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
        """
        # Log to file
        msg = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(msg)
        
        # Log to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def close(self):
        """Close logger and TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
        self.info("Logger closed")


# Test function
if __name__ == "__main__":
    # Test logger
    logger = Logger(exp_name="test_experiment", use_tensorboard=True)
    
    logger.info("Testing logger functionality")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    
    # Test metrics logging
    for step in range(5):
        metrics = {
            'loss': 1.0 / (step + 1),
            'accuracy': 0.5 + step * 0.1,
            'communication_time': 10 - step
        }
        logger.log_metrics(metrics, step)
    
    logger.info("Test completed successfully!")
    logger.close()
    
    print(f"\nLog files saved to: {logger.log_dir}")
