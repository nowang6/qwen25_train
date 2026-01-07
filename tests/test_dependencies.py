"""Tests for dependency installation and import."""

import pytest
import sys


@pytest.mark.dependency
class TestCoreDependencies:
    """Tests for core dependencies."""
    
    def test_torch_import(self):
        """Test that torch can be imported."""
        try:
            import torch
            assert torch is not None
        except ImportError as e:
            pytest.fail(f"Failed to import torch: {e}")
    
    def test_torch_version(self):
        """Test that torch version is >= 2.0.0."""
        import torch
        from packaging import version
        
        assert version.parse(torch.__version__) >= version.parse("2.0.0"), \
            f"torch version {torch.__version__} is < 2.0.0"
    
    def test_torch_cuda_available(self):
        """Test if CUDA is available (optional)."""
        import torch
        # This test should not fail if CUDA is not available
        # It's just informational
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            assert torch.cuda.device_count() > 0
    
    def test_transformers_import(self):
        """Test that transformers can be imported."""
        try:
            import transformers
            assert transformers is not None
        except ImportError as e:
            pytest.fail(f"Failed to import transformers: {e}")
    
    def test_transformers_version(self):
        """Test that transformers version is >= 4.35.0."""
        import transformers
        from packaging import version
        
        assert version.parse(transformers.__version__) >= version.parse("4.35.0"), \
            f"transformers version {transformers.__version__} is < 4.35.0"


@pytest.mark.dependency
class TestDataDependencies:
    """Tests for data processing dependencies."""
    
    def test_datasets_import(self):
        """Test that datasets can be imported."""
        try:
            import datasets
            assert datasets is not None
        except ImportError as e:
            pytest.fail(f"Failed to import datasets: {e}")
    
    def test_datasets_version(self):
        """Test that datasets version is >= 2.14.0."""
        import datasets
        from packaging import version
        
        assert version.parse(datasets.__version__) >= version.parse("2.14.0"), \
            f"datasets version {datasets.__version__} is < 2.14.0"


@pytest.mark.dependency
class TestTrainingDependencies:
    """Tests for training dependencies."""
    
    def test_accelerate_import(self):
        """Test that accelerate can be imported."""
        try:
            import accelerate
            assert accelerate is not None
        except ImportError as e:
            pytest.fail(f"Failed to import accelerate: {e}")
    
    def test_accelerate_version(self):
        """Test that accelerate version is >= 0.24.0."""
        import accelerate
        from packaging import version
        
        assert version.parse(accelerate.__version__) >= version.parse("0.24.0"), \
            f"accelerate version {accelerate.__version__} is < 0.24.0"
    
    def test_trl_import(self):
        """Test that trl can be imported."""
        try:
            import trl
            assert trl is not None
        except ImportError as e:
            pytest.fail(f"Failed to import trl: {e}")
    
    def test_peft_import(self):
        """Test that peft can be imported."""
        try:
            import peft
            assert peft is not None
        except ImportError as e:
            pytest.fail(f"Failed to import peft: {e}")


@pytest.mark.dependency
class TestUtilityDependencies:
    """Tests for utility dependencies."""
    
    def test_yaml_import(self):
        """Test that yaml can be imported."""
        try:
            import yaml
            assert yaml is not None
        except ImportError as e:
            pytest.fail(f"Failed to import yaml: {e}")
    
    def test_tqdm_import(self):
        """Test that tqdm can be imported."""
        try:
            from tqdm import tqdm
            assert tqdm is not None
        except ImportError as e:
            pytest.fail(f"Failed to import tqdm: {e}")
    
    def test_tensorboard_import(self):
        """Test that tensorboard can be imported."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            assert SummaryWriter is not None
        except ImportError as e:
            pytest.fail(f"Failed to import tensorboard: {e}")
    
    def test_wandb_import(self):
        """Test that wandb can be imported."""
        try:
            import wandb
            assert wandb is not None
        except ImportError as e:
            pytest.fail(f"Failed to import wandb: {e}")


class TestModuleImports:
    """Tests for local module imports."""
    
    def test_config_import(self):
        """Test that config module can be imported."""
        try:
            from config import TrainingConfig
            assert TrainingConfig is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config module: {e}")
    
    def test_data_loader_import(self):
        """Test that data_loader module can be imported."""
        try:
            from data_loader import load_alpaca_data, validate_alpaca_data
            assert load_alpaca_data is not None
            assert validate_alpaca_data is not None
        except ImportError as e:
            pytest.fail(f"Failed to import data_loader module: {e}")
    
    def test_utils_import(self):
        """Test that utils module can be imported."""
        try:
            from utils import setup_logging, MetricsLogger
            assert setup_logging is not None
            assert MetricsLogger is not None
        except ImportError as e:
            pytest.fail(f"Failed to import utils module: {e}")
    
    def test_trainer_import(self):
        """Test that trainer module can be imported."""
        try:
            from trainer import Trainer
            assert Trainer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import trainer module: {e}")
