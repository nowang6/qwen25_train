# 快速入门：Qwen2.5-0.5B-Instruct SFT训练

**分支**: `001-qwen-sft-training`
**日期**: 2026-01-07

## 概述

本指南帮助您在15分钟内设置环境并启动第一次Qwen2.5-0.5B-Instruct模型的有监督微调训练。

## 前置要求

### 硬件要求
- GPU: NVIDIA GPU，至少10GB显存（如RTX 3080/4090, A100等）
- 内存: 至少8GB系统内存
- 存储: 至少20GB可用空间

### 软件要求
- 操作系统: Linux (Ubuntu 20.04+), macOS, 或 Windows (WSL2推荐)
- Python: 3.9 或更高版本
- CUDA: 11.7 或更高版本（仅GPU训练需要）

## 5分钟快速开始

### 步骤1: 克隆代码库
```bash
# 克隆包含训练代码的仓库
git clone <repository-url>
cd qwen25_train
```

### 步骤2: 安装依赖
```bash
# 使用uv安装Python依赖（推荐）
uv venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

uv pip install -e .
```

### 步骤3: 准备训练数据
创建Alpaca格式的训练数据文件 `data/train_alpaca.json`:
```json
[
  {
    "instruction": "解释机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支，专注于开发算法和统计模型，使计算机系统能够通过经验自动改进性能，而无需显式编程。"
  },
  {
    "instruction": "将以下英文翻译成中文",
    "input": "Hello, how are you?",
    "output": "你好，最近怎么样？"
  }
  // 添加更多样本...
]
```

### 步骤4: 创建配置文件
创建 `config.json`:
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "main"
  },
  "data": {
    "train_file": "data/train_alpaca.json",
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
```

### 步骤5: 启动训练
```bash
python train.py train --config config.json
```

## 详细步骤

### 1. 环境设置

#### 使用uv（推荐）
```bash
# 安装uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

#### 使用conda（替代方案）
```bash
conda create -n qwen-sft python=3.9
conda activate qwen-sft
pip install -e .
```

### 2. 依赖安装

项目使用 `pyproject.toml` 管理依赖。核心依赖包括：
- `transformers>=4.35.0`: Hugging Face Transformers库
- `torch>=2.0.0`: PyTorch深度学习框架
- `datasets>=2.14.0`: 数据处理库
- `accelerate>=0.24.0`: 分布式训练支持

安装所有依赖：
```bash
uv pip install -e .
```

### 3. 数据准备

#### 数据格式要求
训练数据必须为Alpaca格式，包含以下字段：
- `instruction` (必需): 任务指令
- `input` (可选): 输入上下文
- `output` (必需): 期望输出

#### 数据验证
```bash
# 验证数据格式
python train.py validate --config config.json --data-only
```

### 4. 配置训练

#### 最小配置示例
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct"
  },
  "data": {
    "train_file": "data/train_alpaca.json"
  },
  "training": {
    "learning_rate": 2e-4,
    "batch_size": 16,
    "num_epochs": 3
  },
  "output": {
    "dir": "./output"
  }
}
```

#### 配置验证
```bash
python train.py validate --config config.json
```

### 5. 启动训练

#### 基本训练
```bash
python train.py train --config config.json
```

#### 监控训练进度
训练过程中会显示：
- 进度条和预计完成时间
- 当前损失值 (loss)
- 学习率 (learning_rate)
- 训练步数 (step) 和轮数 (epoch)

#### 检查点
- 每1000步自动保存检查点
- 检查点保存在 `{output_dir}/checkpoint-{step}/`
- 最多保留3个最新检查点

### 6. 恢复训练

如果训练中断，可以从检查点恢复：
```bash
python train.py train --config config.json --resume ./output/checkpoint-1000
```

### 7. 使用训练后的模型

#### 加载模型进行推理
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-final")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 推理示例
inputs = tokenizer("解释人工智能", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## 故障排除

### 常见问题

#### 1. GPU内存不足
**症状**: CUDA out of memory 错误
**解决方案**:
- 减小 `batch_size` (如从16减到8)
- 启用梯度检查点: `"gradient_checkpointing": true`
- 使用更小的模型

#### 2. 训练速度慢
**解决方案**:
- 确保使用GPU训练 (`device: "cuda"`)
- 使用混合精度训练: `"mixed_precision": "bf16"`
- 增加 `gradient_accumulation_steps` 以使用更大有效批大小

#### 3. 数据加载错误
**症状**: "Invalid data format" 错误
**解决方案**:
- 验证数据文件格式为有效JSON
- 确保每个样本包含 `instruction` 和 `output` 字段
- 运行数据验证: `python train.py validate --config config.json --data-only`

### 调试模式

启用详细输出以调试问题：
```bash
python train.py train --config config.json --verbose
```

## 高级配置

### 多GPU训练
```json
{
  "hardware": {
    "device": "cuda",
    "multi_gpu": true
  }
}
```

### 混合精度训练
```json
{
  "hardware": {
    "mixed_precision": "bf16"  # 或 "fp16"
  }
}
```

### 自定义优化器
```json
{
  "training": {
    "optimizer": "adamw_8bit",  # 8-bit AdamW，节省内存
    "weight_decay": 0.01
  }
}
```

## 性能优化建议

### 内存优化
1. 使用梯度检查点 (`gradient_checkpointing: true`)
2. 使用8-bit优化器 (`optimizer: "adamw_8bit"`)
3. 使用BF16混合精度 (`mixed_precision: "bf16"`)

### 速度优化
1. 使用更大的批大小（在内存允许范围内）
2. 启用Tensor Cores（使用BF16/FP16）
3. 使用数据并行（多GPU）

## 下一步

### 验证训练效果
1. 检查训练损失是否稳定下降
2. 验证模型在测试集上的表现
3. 进行人工评估生成质量

### 扩展功能
1. 添加验证集监控
2. 实现早停机制
3. 添加学习率调度
4. 支持LoRA等参数高效微调方法

## 获取帮助

- 查看详细文档: `docs/` 目录
- 报告问题: GitHub Issues
- 社区支持: Hugging Face论坛

## 预计时间线

| 步骤 | 预计时间 |
|------|----------|
| 环境设置 | 2-5分钟 |
| 数据准备 | 5-10分钟 |
| 首次训练启动 | 1-2分钟 |
| 完整训练时间 | 数小时到数天（取决于数据大小） |

---

**提示**: 首次运行时建议使用小数据集（如100条样本）进行测试，验证整个流程正常工作后再使用完整数据集。