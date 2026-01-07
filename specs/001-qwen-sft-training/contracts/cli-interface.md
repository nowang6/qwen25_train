# CLI接口合同：Qwen2.5 SFT训练工具

**分支**: `001-qwen-sft-training`
**日期**: 2026-01-07

## 概述

命令行接口用于启动和管理Qwen2.5-0.5B-Instruct模型的有监督微调训练。

## 命令结构

```bash
python train.py [命令] [选项]
```

### 主要命令

#### 1. `train` - 启动训练
启动有监督微调训练过程。

**用法**:
```bash
python train.py train --config <配置文件> [选项]
```

**选项**:
| 选项 | 缩写 | 类型 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--config` | `-c` | 文件路径 | 必填 | 训练配置文件路径 |
| `--resume` | `-r` | 检查点路径 | 无 | 从检查点恢复训练 |
| `--device` | `-d` | 字符串 | "cuda" | 训练设备："cuda"或"cpu" |
| `--verbose` | `-v` | 标志 | False | 详细输出模式 |

**示例**:
```bash
# 使用配置文件启动训练
python train.py train --config config.json

# 从检查点恢复训练
python train.py train --config config.json --resume ./output/checkpoint-1000

# 在CPU上训练（测试用）
python train.py train --config config.json --device cpu
```

#### 2. `validate` - 验证配置和数据
验证训练配置和数据文件格式。

**用法**:
```bash
python train.py validate --config <配置文件>
```

**选项**:
| 选项 | 缩写 | 类型 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--config` | `-c` | 文件路径 | 必填 | 训练配置文件路径 |
| `--data-only` |  | 标志 | False | 仅验证数据文件 |
| `--config-only` |  | 标志 | False | 仅验证配置文件 |

**示例**:
```bash
# 验证配置和数据
python train.py validate --config config.json

# 仅验证数据文件
python train.py validate --config config.json --data-only
```

#### 3. `export` - 导出模型
将训练后的模型导出为推理格式。

**用法**:
```bash
python train.py export --checkpoint <检查点路径> --output <输出路径>
```

**选项**:
| 选项 | 缩写 | 类型 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--checkpoint` | `-i` | 目录路径 | 必填 | 检查点目录路径 |
| `--output` | `-o` | 目录路径 | 必填 | 输出目录路径 |
| `--format` | `-f` | 字符串 | "pytorch" | 导出格式："pytorch"或"safetensors" |

**示例**:
```bash
# 导出为PyTorch格式
python train.py export --checkpoint ./output/checkpoint-final --output ./exported-model
```

## 配置文件格式

配置文件为JSON格式，包含所有训练参数。

**位置**: 用户指定路径（通过`--config`选项）

**示例配置文件** (`config.json`):
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "revision": "main"
  },
  "data": {
    "train_file": "./data/train_alpaca.json",
    "validation_file": null,
    "max_length": 2048,
    "format": "alpaca"
  },
  "training": {
    "learning_rate": 2e-4,
    "batch_size": 16,
    "num_epochs": 3,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 1000,
    "eval_steps": null,
    "optimizer": "adamw",
    "scheduler": "linear"
  },
  "output": {
    "dir": "./output",
    "save_total_limit": 3,
    "overwrite_output_dir": false
  },
  "hardware": {
    "device": "cuda",
    "mixed_precision": "bf16",
    "gradient_checkpointing": false
  }
}
```

### 配置验证规则

1. **必需字段**:
   - `model.name`: 必须为字符串
   - `data.train_file`: 必须为存在的文件路径
   - `training.learning_rate`: 必须在1e-5到1e-3之间
   - `training.batch_size`: 必须为正整数
   - `training.num_epochs`: 必须为正整数

2. **可选字段**:
   - `data.validation_file`: 可为null或文件路径
   - `data.max_length`: 默认2048
   - `training.gradient_accumulation_steps`: 默认1
   - `output.save_total_limit`: 默认3

## 输出规范

### 标准输出流

1. **STDOUT**: 训练进度信息（人类可读）
   - 进度条
   - 关键指标（loss, learning_rate）
   - 训练状态更新

2. **STDERR**: 错误和警告信息
   - 验证错误
   - 运行时错误
   - 警告信息

### 输出文件结构

训练输出目录结构：
```
输出目录/
├── config.json              # 训练配置副本
├── metrics.jsonl            # 训练指标记录
├── trainer_state.json       # 训练器状态
├── checkpoint-1000/         # 检查点目录
│   ├── pytorch_model.bin
│   ├── optimizer.pt
│   ├── training_args.bin
│   └── trainer_state.json
├── checkpoint-2000/
└── ...
```

## 退出代码

| 代码 | 含义 | 描述 |
|------|------|------|
| 0 | 成功 | 命令成功完成 |
| 1 | 配置错误 | 配置文件或参数无效 |
| 2 | 数据错误 | 训练数据格式错误或不可读 |
| 3 | 运行时错误 | 训练过程中发生错误 |
| 4 | 设备错误 | GPU/CPU设备问题 |
| 5 | 内存错误 | 内存不足 |
| 10 | 恢复错误 | 从检查点恢复失败 |

## 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 可见GPU设备 | 所有可用 |
| `TRANSFORMERS_CACHE` | Transformers缓存目录 | 系统默认 |
| `HF_HOME` | Hugging Face缓存目录 | 系统默认 |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA内存配置 | 未设置 |

## 错误处理

### 输入验证错误
- **症状**: 命令立即失败，显示错误信息
- **处理**: 验证所有输入参数和文件，提供清晰错误信息
- **示例**: "错误: 训练数据文件不存在: /path/to/data.json"

### 运行时错误
- **症状**: 训练过程中失败
- **处理**: 尝试保存检查点，提供错误上下文
- **示例**: "错误: GPU内存不足，尝试减小batch_size"

### 恢复机制
- 训练中断时，可以从最新检查点恢复
- 恢复命令: `python train.py train --config config.json --resume <最新检查点>`

## 日志记录

### 日志级别
1. **INFO** (默认): 训练进度、检查点保存
2. **DEBUG** (`--verbose`): 详细调试信息
3. **WARNING**: 非致命问题警告
4. **ERROR**: 错误信息

### 日志输出
- 控制台: 人类可读格式
- 文件: JSON格式（metrics.jsonl），便于分析

## 性能要求

### 响应时间
- 配置验证: < 1秒
- 数据加载: < 30秒（10万条样本）
- 检查点保存: < 60秒

### 资源使用
- GPU内存: 10-15GB
- CPU内存: 4-8GB
- 磁盘空间: 每个检查点2-3GB

## 兼容性

### Python版本
- 最低: Python 3.9
- 推荐: Python 3.10+

### 操作系统
- Linux (主要支持)
- macOS (CPU训练)
- Windows (有限支持)

### CUDA版本
- CUDA 11.7+ (推荐)
- 支持CPU训练模式