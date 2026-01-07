"""Configuration management for Qwen2.5 SFT training."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 2e-4
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 2048
    checkpoint_steps: int = 1000
    output_dir: str = "./output"
    data_file: str = "./data/train_alpaca.json"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    device: str = "cuda"
    mixed_precision: Optional[str] = None
    gradient_checkpointing: bool = False
    # Wandb configuration
    use_wandb: bool = True
    wandb_project: str = "qwen25-sft"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not (1e-5 <= self.learning_rate <= 1e-3):
            raise ValueError(
                f"learning_rate must be between 1e-5 and 1e-3, got {self.learning_rate}"
            )
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        
        if self.checkpoint_steps <= 0:
            raise ValueError(f"checkpoint_steps must be positive, got {self.checkpoint_steps}")
        
        if not Path(self.output_dir).parent.exists():
            logger.warning(f"Output directory parent does not exist: {self.output_dir}")
        
        if not Path(self.data_file).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")
        
        if self.mixed_precision not in [None, "bf16", "fp16"]:
            raise ValueError(f"mixed_precision must be None, 'bf16', or 'fp16', got {self.mixed_precision}")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Flatten nested config structure
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_config[sub_key] = sub_value
            else:
                flat_config[key] = value
        return cls(**flat_config)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = cls.from_dict(config_dict)
        logger.info(f"Configuration loaded from {path}")
        return config

