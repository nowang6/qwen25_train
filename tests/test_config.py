"""Tests for configuration validation."""

import pytest
import json
import tempfile
from pathlib import Path

from config import TrainingConfig


@pytest.fixture
def temp_config_file():
    """Create temporary config file for testing."""
    config_dict = {
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct"
        },
        "data": {
            "train_file": "./data/train_alpaca.json",
            "max_length": 2048
        },
        "training": {
            "learning_rate": 2e-4,
            "batch_size": 16,
            "num_epochs": 3,
            "save_steps": 1000
        },
        "output": {
            "dir": "./output"
        }
    }
    
    # Create temporary data file
    data_file = Path("./data/train_alpaca.json")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(data_file, 'w') as f:
        json.dump([{"instruction": "test", "input": "", "output": "output"}], f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink()
    if data_file.exists():
        data_file.unlink()


class TestTrainingConfig:
    """Tests for TrainingConfig class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig(
            data_file="./data/test.json"
        )
        assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.learning_rate == 2e-4
        assert config.batch_size == 16
        assert config.num_epochs == 3
        assert config.max_length == 2048
        assert config.checkpoint_steps == 1000
        assert config.output_dir == "./output"
    
    def test_custom_values(self):
        """Test that custom values are set correctly."""
        config = TrainingConfig(
            model_name="custom-model",
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=5,
            max_length=1024,
            checkpoint_steps=500,
            output_dir="./custom_output",
            data_file="./data/test.json"
        )
        assert config.model_name == "custom-model"
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 5
        assert config.max_length == 1024
        assert config.checkpoint_steps == 500
        assert config.output_dir == "./custom_output"
    
    def test_learning_rate_validation(self, temp_config_file):
        """Test learning rate validation."""
        # Valid learning rates
        for lr in [1e-5, 1e-4, 2e-4, 1e-3]:
            config = TrainingConfig(learning_rate=lr, data_file="./data/test.json")
            assert config.learning_rate == lr
        
        # Invalid learning rates
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=1e-6, data_file="./data/test.json")
        
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=1e-2, data_file="./data/test.json")
    
    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch sizes
        for bs in [1, 8, 16, 32, 64]:
            config = TrainingConfig(batch_size=bs, data_file="./data/test.json")
            assert config.batch_size == bs
        
        # Invalid batch sizes
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0, data_file="./data/test.json")
        
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=-1, data_file="./data/test.json")
    
    def test_num_epochs_validation(self):
        """Test num_epochs validation."""
        # Valid num_epochs
        for ne in [1, 2, 3, 5, 10]:
            config = TrainingConfig(num_epochs=ne, data_file="./data/test.json")
            assert config.num_epochs == ne
        
        # Invalid num_epochs
        with pytest.raises(ValueError, match="num_epochs"):
            TrainingConfig(num_epochs=0, data_file="./data/test.json")
        
        with pytest.raises(ValueError, match="num_epochs"):
            TrainingConfig(num_epochs=-1, data_file="./data/test.json")
    
    def test_max_length_validation(self):
        """Test max_length validation."""
        # Valid max_length
        for ml in [512, 1024, 2048, 4096]:
            config = TrainingConfig(max_length=ml, data_file="./data/test.json")
            assert config.max_length == ml
        
        # Invalid max_length
        with pytest.raises(ValueError, match="max_length"):
            TrainingConfig(max_length=0, data_file="./data/test.json")
        
        with pytest.raises(ValueError, match="max_length"):
            TrainingConfig(max_length=-1, data_file="./data/test.json")
    
    def test_checkpoint_steps_validation(self):
        """Test checkpoint_steps validation."""
        # Valid checkpoint_steps
        for cs in [100, 500, 1000, 2000]:
            config = TrainingConfig(checkpoint_steps=cs, data_file="./data/test.json")
            assert config.checkpoint_steps == cs
        
        # Invalid checkpoint_steps
        with pytest.raises(ValueError, match="checkpoint_steps"):
            TrainingConfig(checkpoint_steps=0, data_file="./data/test.json")
        
        with pytest.raises(ValueError, match="checkpoint_steps"):
            TrainingConfig(checkpoint_steps=-1, data_file="./data/test.json")
    
    def test_device_validation(self):
        """Test device validation."""
        # Valid devices
        for device in ["cuda", "cpu"]:
            config = TrainingConfig(device=device, data_file="./data/test.json")
            assert config.device == device
        
        # Invalid device
        with pytest.raises(ValueError, match="device"):
            TrainingConfig(device="invalid", data_file="./data/test.json")
    
    def test_mixed_precision_validation(self):
        """Test mixed_precision validation."""
        # Valid mixed_precision
        for mp in [None, "bf16", "fp16"]:
            config = TrainingConfig(mixed_precision=mp, data_file="./data/test.json")
            assert config.mixed_precision == mp
        
        # Invalid mixed_precision
        with pytest.raises(ValueError, match="mixed_precision"):
            TrainingConfig(mixed_precision="invalid", data_file="./data/test.json")
    
    def test_data_file_not_found(self):
        """Test that missing data file raises error."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            TrainingConfig(data_file="./nonexistent/file.json")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(
            model_name="test-model",
            data_file="./data/test.json"
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["data_file"] == "./data/test.json"
    
    def test_save_and_load(self, temp_config_file):
        """Test saving and loading configuration."""
        config = TrainingConfig.load(temp_config_file)
        assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
        assert config.learning_rate == 2e-4
        assert config.batch_size == 16
        assert config.num_epochs == 3
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model_name": "test-model",
            "learning_rate": 1e-4,
            "data_file": "./data/test.json"
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.model_name == "test-model"
        assert config.learning_rate == 1e-4
    
    def test_from_dict_nested(self):
        """Test creating config from nested dictionary."""
        config_dict = {
            "model": {"name": "test-model"},
            "training": {"learning_rate": 1e-4},
            "data_file": "./data/test.json"
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.model_name == "test-model"
        assert config.learning_rate == 1e-4
