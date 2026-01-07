# 数据模型：Qwen2.5-0.5B-Instruct SFT训练

**分支**: `001-qwen-sft-training`
**日期**: 2026-01-07

## 实体定义

### 1. 训练配置 (TrainingConfig)

训练超参数和设置，控制训练过程的行为。

| 字段 | 类型 | 描述 | 约束/验证 |
|------|------|------|-----------|
| model_name | string | 模型名称/标识符 | 必须为"Qwen/Qwen2.5-0.5B-Instruct" |
| learning_rate | float | 学习率 | 范围：1e-5 到 1e-3，默认：2e-4 |
| batch_size | integer | 批大小 | 正整數，默认：16，基于GPU内存调整 |
| num_epochs | integer | 训练轮数 | 正整數，默认：3 |
| max_length | integer | 序列最大长度 | 正整數，默认：2048（模型最大长度） |
| checkpoint_steps | integer | 检查点保存步数间隔 | 正整數，默认：1000 |
| output_dir | string | 输出目录路径 | 有效文件系统路径 |
| data_file | string | 训练数据文件路径 | 有效文件路径，必须存在 |

**关系**:
- 一个训练配置对应一次训练运行
- 训练配置生成多个模型检查点

### 2. 训练数据样本 (TrainingSample)

Alpaca格式的指令跟随数据样本。

| 字段 | 类型 | 描述 | 约束/验证 |
|------|------|------|-----------|
| instruction | string | 任务指令 | 非空字符串 |
| input | string | 可选输入上下文 | 可为空字符串 |
| output | string | 期望输出 | 非空字符串 |

**关系**:
- 多个训练样本组成训练数据集
- 训练数据集被训练过程使用

### 3. 模型检查点 (ModelCheckpoint)

训练过程中保存的模型状态。

| 字段 | 类型 | 描述 | 约束/验证 |
|------|------|------|-----------|
| checkpoint_id | string | 检查点标识符 | 格式："checkpoint-{step}" |
| step | integer | 训练步数 | 非负整数 |
| epoch | float | 训练轮数 | 非负浮点数 |
| model_state | binary | 模型权重 | PyTorch模型状态字典 |
| optimizer_state | binary | 优化器状态 | PyTorch优化器状态 |
| training_config | object | 训练配置快照 | 训练时的配置副本 |
| timestamp | datetime | 保存时间 | ISO 8601格式 |

**关系**:
- 一个检查点属于一次训练运行
- 可以从检查点恢复训练或加载推理

### 4. 训练指标 (TrainingMetrics)

训练过程中记录的性能指标。

| 字段 | 类型 | 描述 | 约束/验证 |
|------|------|------|-----------|
| step | integer | 训练步数 | 非负整数 |
| loss | float | 损失值 | 非负浮点数 |
| learning_rate | float | 当前学习率 | 非负浮点数 |
| gradient_norm | float | 梯度范数 | 非负浮点数 |
| epoch | float | 当前轮数 | 非负浮点数 |
| samples_per_second | float | 每秒处理样本数 | 非负浮点数 |

**关系**:
- 多个指标点组成训练历史
- 指标属于一次训练运行

## 状态转换

### 训练过程状态机

```
待启动 (Idle)
    ↓
数据加载中 (LoadingData)
    ↓
模型初始化中 (InitializingModel)
    ↓
训练中 (Training) → 检查点保存中 (SavingCheckpoint) → 训练中 (Training)
    ↓
完成 (Completed)
    ↓
错误 (Error) [可从检查点恢复]
```

### 检查点生命周期

```
创建中 (Creating) → 可用 (Available) → 加载中 (Loading) → 使用中 (InUse)
                                ↓
                                删除 (Deleted)
```

## 数据验证规则

### 训练数据验证
1. JSON文件必须包含对象数组
2. 每个对象必须包含`instruction`和`output`字段
3. `instruction`和`output`必须是非空字符串
4. `input`字段可选，如果存在必须是字符串

### 训练配置验证
1. `learning_rate`必须在有效范围内（1e-5到1e-3）
2. `batch_size`必须是正整数
3. `num_epochs`必须是正整数
4. `output_dir`必须是可写目录路径
5. `data_file`必须是可读文件路径

### 模型检查点验证
1. 检查点文件必须包含所有必需字段
2. 模型状态必须与当前模型架构兼容
3. 优化器状态必须与当前优化器配置兼容

## 持久化格式

### 训练配置
- **格式**: JSON文件（config.json）
- **位置**: 输出目录根层级
- **示例**:
```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "learning_rate": 0.0002,
  "batch_size": 16,
  "num_epochs": 3,
  "max_length": 2048,
  "checkpoint_steps": 1000,
  "output_dir": "./output",
  "data_file": "./data/train_alpaca.json"
}
```

### 训练数据
- **格式**: JSON行文件（.jsonl）或JSON数组
- **位置**: 用户指定路径
- **示例**:
```json
[
  {
    "instruction": "解释机器学习",
    "input": "",
    "output": "机器学习是..."
  }
]
```

### 模型检查点
- **格式**: PyTorch检查点目录
- **位置**: `{output_dir}/checkpoint-{step}/`
- **内容**:
  - `pytorch_model.bin` - 模型权重
  - `optimizer.pt` - 优化器状态
  - `training_args.bin` - 训练配置
  - `trainer_state.json` - 训练状态

### 训练指标
- **格式**: JSON行文件（metrics.jsonl）
- **位置**: `{output_dir}/`
- **示例**:
```json
{"step": 100, "loss": 2.34, "learning_rate": 0.0002, "gradient_norm": 1.23, "epoch": 0.5}
```

## 数据流

```
训练数据文件 (Alpaca JSON)
        ↓
数据加载器 (验证 + 预处理)
        ↓
训练循环 (使用配置)
        ↓
模型检查点 (定期保存)
        ↓
训练指标 (实时记录)
```

## 约束和限制

1. **数据大小**: 训练数据集应适合内存（数万条样本）
2. **检查点存储**: 每个检查点约2-3GB（模型大小 + 优化器状态）
3. **内存使用**: 训练需要10-15GB GPU内存
4. **文件数量**: 遵守5个文件限制
5. **类数量**: 遵守10个类限制