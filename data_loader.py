"""Data loading and validation for Alpaca format training data."""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_alpaca_sample(sample: Dict[str, Any]) -> None:
    """Validate a single Alpaca format sample.
    
    Args:
        sample: Dictionary containing instruction, input, output fields
    
    Raises:
        DataValidationError: If sample is invalid
    """
    if not isinstance(sample, dict):
        raise DataValidationError(f"Sample must be a dictionary, got {type(sample)}")
    
    if "instruction" not in sample:
        raise DataValidationError("Sample missing required field: 'instruction'")
    
    if "output" not in sample:
        raise DataValidationError("Sample missing required field: 'output'")
    
    if not isinstance(sample["instruction"], str) or not sample["instruction"].strip():
        raise DataValidationError("Sample 'instruction' must be a non-empty string")
    
    if not isinstance(sample["output"], str) or not sample["output"].strip():
        raise DataValidationError("Sample 'output' must be a non-empty string")
    
    if "input" in sample and sample["input"] is not None:
        if not isinstance(sample["input"], str):
            raise DataValidationError("Sample 'input' must be a string if provided")


def validate_alpaca_data(data: List[Dict[str, Any]]) -> None:
    """Validate entire Alpaca format dataset.
    
    Args:
        data: List of training samples
    
    Raises:
        DataValidationError: If data is invalid
    """
    if not isinstance(data, list):
        raise DataValidationError(f"Data must be a list, got {type(data)}")
    
    if len(data) == 0:
        raise DataValidationError("Data list cannot be empty")
    
    for i, sample in enumerate(data):
        try:
            validate_alpaca_sample(sample)
        except DataValidationError as e:
            raise DataValidationError(f"Sample {i}: {str(e)}")
    
    logger.info(f"Validated {len(data)} training samples")


def load_alpaca_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate Alpaca format training data.
    
    Args:
        file_path: Path to JSON data file
    
    Returns:
        List of validated training samples
    
    Raises:
        FileNotFoundError: If file doesn't exist
        DataValidationError: If data format is invalid
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    validate_alpaca_data(data)
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    
    return data


def format_alpaca_for_training(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format a single Alpaca sample for chat template.
    
    Args:
        sample: Dictionary with instruction, input, output fields
    
    Returns:
        List of messages in chat format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    instruction = sample["instruction"]
    input_text = sample.get("input", "")
    output = sample["output"]
    
    # Combine instruction and input for user message
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    # Return messages in chat format
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
