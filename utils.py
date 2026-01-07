"""Utility functions for training, logging, and model loading."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
    
    Returns:
        Configured logger
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    return logging.getLogger("qwen25_train")


class MetricsLogger:
    """Log training metrics to file."""
    
    def __init__(self, output_dir: str, filename: str = "metrics.jsonl"):
        """Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics
            filename: Metrics filename
        """
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + "\n")
    
    def read_history(self) -> list[Dict[str, Any]]:
        """Read all logged metrics.
        
        Returns:
            List of metric dictionaries
        """
        if not self.metrics_file.exists():
            return []
        
        history = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        
        return history


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    step: int,
    epoch: float,
    config: Dict[str, Any]
) -> None:
    """Save training checkpoint.
    
    Args:
        model: PyTorch model
        tokenizer: Hugging Face tokenizer
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        checkpoint_dir: Directory to save checkpoint
        step: Current training step
        epoch: Current epoch
        config: Training configuration
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(str(checkpoint_path))
    tokenizer.save_pretrained(str(checkpoint_path))
    
    # Save optimizer state
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "config": config
    }, checkpoint_path / "training_state.bin")
    
    logging.getLogger(__name__).info(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    model,
    optimizer,
    scheduler
) -> tuple:
    """Load training checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
    
    Returns:
        Tuple of (step, epoch)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # Load model weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path))
    
    # Load training state
    training_state = torch.load(checkpoint_path / "training_state.bin")
    
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    
    if scheduler and training_state["scheduler_state_dict"]:
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
    
    step = training_state["step"]
    epoch = training_state["epoch"]
    
    logging.getLogger(__name__).info(f"Checkpoint loaded from {checkpoint_dir}, step={step}, epoch={epoch}")
    
    return step, epoch


def check_dependencies() -> Dict[str, Any]:
    """Check if all required dependencies are installed and compatible.
    
    Returns:
        Dictionary with dependency check results
    """
    logger = logging.getLogger(__name__)
    results = {
        "python_version": sys.version,
        "dependencies": {},
        "all_satisfied": True
    }
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        results["all_satisfied"] = False
        logger.error(f"Python version {python_version.major}.{python_version.minor} is not supported. Required: >=3.9")
    else:
        logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check dependencies
    dependencies = {
        "torch": {"min_version": "2.0.0", "optional": False},
        "transformers": {"min_version": "4.35.0", "optional": False},
        "datasets": {"min_version": "2.14.0", "optional": False},
        "accelerate": {"min_version": "0.24.0", "optional": False},
        "trl": {"min_version": "0.7.0", "optional": False},
        "peft": {"min_version": "0.6.0", "optional": False},
    }
    
    from packaging import version
    
    for package_name, requirements in dependencies.items():
        try:
            module = __import__(package_name)
            installed_version = getattr(module, "__version__", "unknown")
            min_version = requirements["min_version"]
            
            version_satisfied = version.parse(installed_version) >= version.parse(min_version)
            
            results["dependencies"][package_name] = {
                "installed": True,
                "version": installed_version,
                "required_version": min_version,
                "satisfied": version_satisfied
            }
            
            if version_satisfied:
                logger.info(f"✓ {package_name}: {installed_version} (>= {min_version})")
            else:
                results["all_satisfied"] = False
                logger.error(f"✗ {package_name}: {installed_version} (< {min_version})")
        
        except ImportError:
            results["dependencies"][package_name] = {
                "installed": False,
                "version": None,
                "required_version": min_version,
                "satisfied": False
            }
            
            if not requirements["optional"]:
                results["all_satisfied"] = False
                logger.error(f"✗ {package_name}: Not installed (required: >= {min_version})")
            else:
                logger.warning(f"⚠ {package_name}: Not installed (optional)")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        results["cuda_available"] = cuda_available
        
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available: {cuda_version}, {gpu_count} GPU(s) detected")
            results["cuda_version"] = cuda_version
            results["gpu_count"] = gpu_count
        else:
            logger.warning("CUDA not available. Training will run on CPU (slow).")
    except ImportError:
        logger.warning("Cannot check CUDA availability (torch not imported)")
        results["cuda_available"] = False
    
    return results


def validate_environment(silent: bool = False) -> bool:
    """Validate that the training environment is properly configured.
    
    Args:
        silent: If True, suppress output
    
    Returns:
        True if environment is valid, False otherwise
    """
    if not silent:
        print("Validating training environment...")
        print("-" * 50)
    
    results = check_dependencies()
    
    if not silent:
        print("-" * 50)
        if results["all_satisfied"]:
            print("✓ All dependencies satisfied!")
            return True
        else:
            print("✗ Some dependencies are missing or incompatible.")
            print("\nPlease install missing dependencies:")
            print("  uv pip install -e .")
            return False
    
    return results["all_satisfied"]
