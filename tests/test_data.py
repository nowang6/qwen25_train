"""Tests for data loading and validation."""

import pytest
import json
import tempfile
from pathlib import Path

from data_loader import (
    validate_alpaca_sample,
    validate_alpaca_data,
    load_alpaca_data,
    format_alpaca_for_training,
    DataValidationError
)


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


class TestValidateAlpacaSample:
    """Tests for validate_alpaca_sample function."""
    
    def test_valid_sample(self):
        """Test validation of a valid sample."""
        sample = {
            "instruction": "测试指令",
            "input": "",
            "output": "测试输出"
        }
        validate_alpaca_sample(sample)  # Should not raise
    
    def test_missing_instruction(self):
        """Test that missing instruction raises error."""
        sample = {
            "input": "",
            "output": "测试输出"
        }
        with pytest.raises(DataValidationError, match="instruction"):
            validate_alpaca_sample(sample)
    
    def test_missing_output(self):
        """Test that missing output raises error."""
        sample = {
            "instruction": "测试指令",
            "input": ""
        }
        with pytest.raises(DataValidationError, match="output"):
            validate_alpaca_sample(sample)
    
    def test_empty_instruction(self):
        """Test that empty instruction raises error."""
        sample = {
            "instruction": "   ",
            "input": "",
            "output": "测试输出"
        }
        with pytest.raises(DataValidationError, match="instruction"):
            validate_alpaca_sample(sample)
    
    def test_empty_output(self):
        """Test that empty output raises error."""
        sample = {
            "instruction": "测试指令",
            "input": "",
            "output": "   "
        }
        with pytest.raises(DataValidationError, match="output"):
            validate_alpaca_sample(sample)
    
    def test_non_string_instruction(self):
        """Test that non-string instruction raises error."""
        sample = {
            "instruction": 123,
            "input": "",
            "output": "测试输出"
        }
        with pytest.raises(DataValidationError, match="instruction"):
            validate_alpaca_sample(sample)
    
    def test_non_dict_input(self):
        """Test that non-dict input raises error."""
        with pytest.raises(DataValidationError, match="dictionary"):
            validate_alpaca_sample("not a dict")


class TestValidateAlpacaData:
    """Tests for validate_alpaca_data function."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        data = [
            {"instruction": "指令1", "input": "", "output": "输出1"},
            {"instruction": "指令2", "input": "输入", "output": "输出2"}
        ]
        validate_alpaca_data(data)  # Should not raise
    
    def test_empty_data(self):
        """Test that empty data raises error."""
        data = []
        with pytest.raises(DataValidationError, match="empty"):
            validate_alpaca_data(data)
    
    def test_non_list_data(self):
        """Test that non-list data raises error."""
        with pytest.raises(DataValidationError, match="list"):
            validate_alpaca_data("not a list")
    
    def test_invalid_sample_in_data(self):
        """Test that invalid sample in data raises error."""
        data = [
            {"instruction": "指令1", "input": "", "output": "输出1"},
            {"instruction": "", "input": "", "output": "输出2"}  # Invalid
        ]
        with pytest.raises(DataValidationError, match="Sample 1"):
            validate_alpaca_data(data)


class TestLoadAlpacaData:
    """Tests for load_alpaca_data function."""
    
    def test_load_valid_file(self, temp_data_file):
        """Test loading a valid data file."""
        data = load_alpaca_data(temp_data_file)
        assert len(data) == 2
        assert data[0]["instruction"] == "解释机器学习"
    
    def test_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_alpaca_data("/nonexistent/file.json")
    
    def test_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_alpaca_data(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_data_format(self, temp_data_file):
        """Test loading file with invalid data format."""
        # Write invalid data to temp file
        with open(temp_data_file, 'w') as f:
            json.dump([{"instruction": "test"}], f)  # Missing output
        
        with pytest.raises(DataValidationError):
            load_alpaca_data(temp_data_file)


class TestFormatAlpacaForTraining:
    """Tests for format_alpaca_for_training function."""
    
    def test_format_without_input(self):
        """Test formatting sample without input."""
        sample = {
            "instruction": "解释机器学习",
            "input": "",
            "output": "机器学习是AI的分支。"
        }
        messages = format_alpaca_for_training(sample)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert sample["instruction"] in messages[0]["content"]
        assert sample["output"] in messages[1]["content"]
        assert "input" not in messages[0]["content"].lower() or messages[0]["content"].count("\n\n") == 0
    
    def test_format_with_input(self):
        """Test formatting sample with input."""
        sample = {
            "instruction": "翻译",
            "input": "Hello",
            "output": "你好"
        }
        messages = format_alpaca_for_training(sample)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert sample["instruction"] in messages[0]["content"]
        assert sample["input"] in messages[0]["content"]
        assert sample["output"] in messages[1]["content"]
