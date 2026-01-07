# Qwen2.5-0.5B-Instruct 有监督微调训练工具

对Qwen2.5-0.5B-Instruct模型进行有监督微调（SFT）的训练工具。

## 功能特性

- 支持Alpaca格式的指令跟随数据
- 完整的训练流程，包括数据加载、模型训练、检查点保存
- 自动依赖管理和环境验证
- 支持GPU/CPU训练
- 混合精度训练支持（BF16/FP16）
- 训练指标实时记录
- 检查点管理（自动保留最新3个检查点）
- 支持从检查点恢复训练

## 快速开始

### 1. 安装依赖

```bash
# 使用uv安装依赖（推荐）
uv pip install -e .

# 或使用pip
pip install -e .
```

### 2. 准备训练数据

创建Alpaca格式的训练数据文件 `data/train_alpaca.json`：

```json
[
  {
    "instruction": "解释机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "翻译成中文",
    "input": "Hello, world!",
    "output": "你好，世界！"
  }
]
```

### 3. 配置训练参数

编辑 `config.json` 文件（示例）：

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "data_file": "./data/train_alpaca.json",
  "learning_rate": 0.0002,
  "batch_size": 16,
  "num_epochs": 3,
  "max_length": 2048,
  "checkpoint_steps": 1000,
  "output_dir": "./output",
  "device": "cuda"
}
```

### 4. 验证配置和数据

```bash
python train.py validate --config config.json
```

### 5. 启动训练

```bash
python train.py train --config config.json
```

## 命令行接口

### 训练命令

```bash
# 基本训练
python train.py train --config config.json

# 从检查点恢复训练
python train.py train --config config.json --resume ./output/checkpoint-1000

# 在CPU上训练（测试用）
python train.py train --config config.json --device cpu

# 详细日志
python train.py train --config config.json --verbose

# 跳过依赖检查
python train.py train --config config.json --skip-dependency-check
```

### 验证命令

```bash
# 验证配置和数据
python train.py validate --config config.json

# 仅验证数据
python train.py validate --config config.json --data-only

# 仅验证配置
python train.py validate --config config.json --config-only

# 检查依赖
python train.py validate --config config.json --check-deps
```

### 导出命令

```bash
python train.py export --checkpoint ./output/checkpoint-final --output ./exported-model
```

## 配置参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `model_name` | string | Qwen/Qwen2.5-0.5B-Instruct | 模型名称 |
| `data_file` | string | - | 训练数据文件路径（必需） |
| `learning_rate` | float | 2e-4 | 学习率（1e-5到1e-3） |
| `batch_size` | int | 16 | 批大小 |
| `num_epochs` | int | 3 | 训练轮数 |
| `max_length` | int | 2048 | 最大序列长度 |
| `checkpoint_steps` | int | 1000 | 检查点保存间隔（步数） |
| `output_dir` | string | ./output | 输出目录 |
| `device` | string | cuda | 训练设备（cuda/cpu） |
| `warmup_steps` | int | 100 | 学习率预热步数 |
| `logging_steps` | int | 10 | 日志记录间隔 |
| `mixed_precision` | string | null | 混合精度（null/bf16/fp16） |
| `gradient_checkpointing` | bool | false | 是否启用梯度检查点 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |
| `save_total_limit` | int | 3 | 保留的检查点数量 |

## 输出目录结构

训练完成后，输出目录结构如下：

```
output/
├── config.json              # 训练配置副本
├── metrics.jsonl            # 训练指标记录
├── checkpoint-1000/         # 检查点目录
│   ├── config.json
│   ├── model.safetensors    # 模型权重
│   ├── training_state.bin   # 训练状态
│   └── ... (tokenizer files)
├── checkpoint-2000/
└── checkpoint-final/        # 最终检查点
```

## 使用训练后的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-final")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 进行推理
inputs = tokenizer("解释人工智能", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_config.py
pytest tests/test_data.py
pytest tests/test_trainer.py

# 显示详细输出
pytest -v tests/

# 显示覆盖率
pytest --cov=.
```

## 故障排除

### GPU内存不足
- 减小 `batch_size`
- 启用 `gradient_checkpointing: true`
- 使用混合精度 `mixed_precision: "bf16"`

### 训练速度慢
- 确保使用GPU训练 `device: "cuda"`
- 使用混合精度训练
- 增加 `batch_size`（在内存允许范围内）

### 数据加载错误
- 验证数据文件格式
- 运行 `python train.py validate --config config.json --data-only`

## 项目结构

```
qwen25_train/
├── config.py              # 配置管理
├── data_loader.py         # 数据加载和验证
├── trainer.py            # 训练循环
├── utils.py              # 工具函数
├── train.py              # 命令行入口
├── pyproject.toml        # 项目依赖配置
├── config.json           # 训练配置示例
├── data/                 # 数据目录
│   └── train_alpaca.json # 训练数据
├── output/               # 输出目录（训练时生成）
├── tests/                # 测试目录
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_trainer.py
│   └── test_dependencies.py
└── README.md             # 本文件
```

## 代码约束

- **核心代码文件数**: 5个（`config.py`, `data_loader.py`, `trainer.py`, `train.py`, `utils.py`）
- **类定义数**: 5个（`TrainingConfig`, `AlpacaDataset`, `Trainer`, `MetricsLogger`, `DataValidationError`）

## 许可证

MIT License
