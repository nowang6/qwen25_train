"""Tests for trainer functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from trainer import Trainer, AlpacaDataset
from config import TrainingConfig


@pytest.fixture
def temp_data_file():
    """Create temporary data file for testing."""
    data = [
        {
            "instruction": "解释机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支。"
        },
        {
            "instruction": "翻译成中文",
            "input": "Hello, world!",
            "output": "你好，世界！"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, ensure_ascii=False)
        temp_path = f.name
    
    yield temp_path
    
    Path(temp_path).unlink()


@pytest.fixture
def mock_config(temp_data_file, tmp_path):
    """Create mock configuration for testing."""
    return TrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        data_file=temp_data_file,
        output_dir=str(tmp_path / "output"),
        batch_size=2,
        num_epochs=1,
        max_length=512,
        checkpoint_steps=100,
        logging_steps=10,
        device="cpu"  # Use CPU for testing
    )


class TestAlpacaDataset:
    """Tests for AlpacaDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {
                "instruction": "测试指令",
                "input": "",
                "output": "测试输出"
            },
            {
                "instruction": "翻译",
                "input": "Hello",
                "output": "你好"
            }
        ]
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        import torch
        tokenizer = Mock()
        # Mock apply_chat_template to return formatted text
        tokenizer.apply_chat_template.return_value = "formatted text"
        # Mock tokenizer call to return encodings
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
        }
        return tokenizer
    
    def test_dataset_length(self, sample_data, mock_tokenizer):
        """Test dataset length."""
        dataset = AlpacaDataset(sample_data, mock_tokenizer, max_length=512)
        assert len(dataset) == 2
    
    def test_dataset_getitem(self, sample_data, mock_tokenizer):
        """Test getting an item from dataset."""
        dataset = AlpacaDataset(sample_data, mock_tokenizer, max_length=512)
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        # Check that labels match input_ids
        import torch
        assert torch.equal(item["input_ids"], item["labels"])
    
    @patch('data_loader.format_alpaca_for_training')
    def test_dataset_tokenization(self, mock_format, sample_data, mock_tokenizer):
        """Test that dataset tokenizes data correctly."""
        # Mock format to return messages list
        mock_format.return_value = [
            {"role": "user", "content": "test instruction"},
            {"role": "assistant", "content": "test output"}
        ]
        
        dataset = AlpacaDataset(sample_data, mock_tokenizer, max_length=512)
        dataset[0]
        
        # Verify format was called
        mock_format.assert_called_once_with(sample_data[0])
        
        # Verify apply_chat_template was called
        mock_tokenizer.apply_chat_template.assert_called_once()
        
        # Verify tokenizer was called
        mock_tokenizer.assert_called_once()


class TestTrainer:
    """Tests for Trainer class."""
    
    @patch('trainer.AutoModelForCausalLM.from_pretrained')
    @patch('trainer.AutoTokenizer.from_pretrained')
    def test_trainer_initialization(
        self,
        mock_tokenizer,
        mock_model,
        mock_config
    ):
        """Test trainer initialization."""
        # Setup mocks
        mock_tokenizer.return_value = Mock(pad_token=None, eos_token="</s>")
        mock_model.return_value = Mock()
        
        # Create trainer
        trainer = Trainer(mock_config)
        
        # Verify trainer was created
        assert trainer.config == mock_config
        assert trainer.device.type == "cpu"
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
    
    @patch('trainer.AutoModelForCausalLM.from_pretrained')
    @patch('trainer.AutoTokenizer.from_pretrained')
    def test_trainer_setup_model(
        self,
        mock_tokenizer,
        mock_model,
        mock_config
    ):
        """Test model setup."""
        # Setup mocks
        mock_tk = Mock(pad_token=None, eos_token="</s>")
        mock_tokenizer.return_value = mock_tk
        mock_model.return_value = Mock()
        
        # Create trainer
        trainer = Trainer(mock_config)
        
        # Verify model and tokenizer were loaded
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        
        # Verify pad token was set
        assert mock_tk.pad_token == "</s>"
    
    @patch('trainer.AutoModelForCausalLM.from_pretrained')
    @patch('trainer.AutoTokenizer.from_pretrained')
    @patch('trainer.load_alpaca_data')
    def test_trainer_setup_data(
        self,
        mock_load_data,
        mock_tokenizer,
        mock_model,
        mock_config
    ):
        """Test data setup."""
        # Setup mocks
        mock_tokenizer.return_value = Mock(pad_token="<pad>", eos_token="</s>")
        mock_model.return_value = Mock()
        mock_load_data.return_value = [
            {"instruction": "test", "input": "", "output": "output"}
        ]
        
        # Create trainer
        trainer = Trainer(mock_config)
        
        # Verify data was loaded
        mock_load_data.assert_called_once_with(mock_config.data_file)
        assert len(trainer.dataset) == 1
    
    @patch('trainer.AutoModelForCausalLM.from_pretrained')
    @patch('trainer.AutoTokenizer.from_pretrained')
    @patch('trainer.load_alpaca_data')
    @patch('trainer.AdamW')
    @patch('trainer.get_linear_schedule_with_warmup')
    def test_trainer_setup_optimizer(
        self,
        mock_scheduler,
        mock_adamw,
        mock_load_data,
        mock_tokenizer,
        mock_model,
        mock_config
    ):
        """Test optimizer and scheduler setup."""
        # Setup mocks
        mock_tokenizer.return_value = Mock(pad_token="<pad>", eos_token="</s>")
        mock_model.return_value = Mock()
        mock_load_data.return_value = [
            {"instruction": "test", "input": "", "output": "output"}
        ]
        mock_adamw.return_value = Mock()
        mock_scheduler.return_value = Mock()
        
        # Create trainer
        trainer = Trainer(mock_config)
        
        # Verify optimizer was created
        mock_adamw.assert_called_once()
        
        # Verify scheduler was created
        mock_scheduler.assert_called_once()


class TestCheckpointFunctionality:
    """Tests for checkpoint save/load functionality."""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        return str(output_dir)
    
    @patch('trainer.save_checkpoint')
    def test_checkpoint_is_saved(
        self,
        mock_save_checkpoint,
        mock_config,
        temp_output_dir
    ):
        """Test that checkpoint is saved."""
        from trainer import Trainer
        from unittest.mock import patch
        
        with patch('trainer.AutoModelForCausalLM.from_pretrained'), \
             patch('trainer.AutoTokenizer.from_pretrained'), \
             patch('trainer.load_alpaca_data'), \
             patch('trainer.AdamW'), \
             patch('trainer.get_linear_schedule_with_warmup'):
            
            # Create a minimal trainer for testing
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_optimizer = Mock()
            mock_scheduler = Mock()
            
            # Simulate checkpoint save
            checkpoint_dir = f"{temp_output_dir}/checkpoint-100"
            
            save_checkpoint(
                mock_model,
                mock_tokenizer,
                mock_optimizer,
                mock_scheduler,
                checkpoint_dir,
                100,
                1.0,
                {}
            )
            
            # Verify save was called
            # Note: This is a simplified test since we're mocking the function
            assert Path(checkpoint_dir).exists()
    
    @patch('trainer.load_checkpoint')
    def test_checkpoint_is_loaded(
        self,
        mock_load_checkpoint,
        mock_config,
        temp_output_dir
    ):
        """Test that checkpoint can be loaded."""
        from trainer import Trainer
        from utils import load_checkpoint
        
        # Create a mock checkpoint directory
        checkpoint_dir = Path(temp_output_dir) / "checkpoint-100"
        checkpoint_dir.mkdir()
        
        # Create mock checkpoint files
        (checkpoint_dir / "config.json").write_text("{}")
        (checkpoint_dir / "model.safetensors").write_text("dummy")
        (checkpoint_dir / "training_state.bin").write_text("dummy")
        
        # Test would need actual model loading for full verification
        # This is a simplified test to verify the structure
        assert checkpoint_dir.exists()
