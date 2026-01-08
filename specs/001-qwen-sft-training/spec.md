# Qwen2.5-0.5B-Instruct SFT 训练需求说明

**分支**: `001-qwen-sft-training`
**创建时间**: 2026-01-07
**目标**: 使用指令跟随数据对 Qwen2.5-0.5B-Instruct 模型进行有监督微调，训练框架为 Transformers，依赖通过 uv 管理，代码简洁（≤5个文件，≤10个类）。

---

## 1. 数据与格式

* **训练数据**: Alpaca JSON 格式

  ```json
  {
    "instruction": "...",
    "input": "...",          // 可选
    "output": "..."
  }
  ```
* **质量要求**: 字段完整，格式正确。
* **额外要求**: 训练过程中需应用大模型对话模板 (`apply_chat_template`) 进行数据封装和输入输出格式化。

---

## 2. 核心功能（User Stories）

| 优先级 | 功能              | 验收标准                                                                                                                      |
| --- | --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| P1  | 指令跟随微调 & 对话模板应用 | 给定符合 Alpaca 格式的数据，启动训练并生成模型检查点，训练过程中显示 loss、learning_rate、gradient_norm、epoch、steps 等指标；训练数据应用 `apply_chat_template` 格式化。 |
| P2  | Python 依赖管理     | 使用 pyproject.toml 安装依赖，确保包版本兼容，可在新环境成功导入关键库。                                                                              |
| P2  | WandB 训练监控（可选）  | 启用 WandB 后自动记录训练指标和超参数，支持可视化；禁用时不影响训练。                                                                                    |
| P3  | 模型保存与加载         | 定期保存完整检查点（模型权重+优化器状态+调度器状态），支持从检查点恢复训练并用于推理。                                                                              |

**边缘情况处理**:

* 数据格式不正确 → 提示缺失字段
* GPU 显存不足 → 提供显存优化选项（BF16/FP16、梯度检查点、梯度累积、动态 padding）
* 训练中断 → 支持从检查点恢复
* WandB 未配置或认证失败 → 提供禁用选项并提示
* BF16 不支持 → 自动降级 FP16/FP32 并警告

---

## 3. 功能需求（FR）

1. **FR-001**: 支持 Alpaca 数据的有监督微调，并应用 `apply_chat_template` 对话模板。
2. **FR-002**: 提供简单训练配置接口（学习率、批大小、轮数），默认值为 Qwen2.5-0.5B-Instruct 推荐值。
3. **FR-003**: 使用 pyproject.toml 管理依赖，保证环境可重复性。
4. **FR-004**: 支持保存/加载模型检查点，训练中断可恢复。
5. **FR-005**: 代码总文件 ≤5，类 ≤10。
6. **FR-006**: 显示并记录训练日志及指标（loss、learning_rate、gradient_norm、epoch、steps）。
7. **FR-007**: 提供显存优化（BF16/FP16、梯度检查点、动态 padding、梯度累积），保证训练效果不受影响。
8. **FR-008**: 可选 WandB 集成（记录指标、超参数、系统资源），禁用不影响训练。

---

## 4. 显存优化策略

**推荐组合（不会影响训练效果）**:

1. **混合精度训练 (BF16)**: 节省约 50% 显存，硬件不支持时降级 FP16/FP32。
2. **梯度检查点**: 激活值显存节省 40-60%，训练时间增加 20-30%。
3. **动态 Padding**: 避免无效 padding，按 batch 内最长序列对齐。
4. **梯度累积**: 小 batch 累积更新，有效 batch size = batch_size × gradient_accumulation_steps。

**推荐 config.json 示例**:

```json
{
  "model_name": "./models/Qwen2.5-0.5B-Instruct",
  "data_file": "./data/train_alpaca.json",
  "learning_rate": 1.0e-5,
  "batch_size": 4,
  "num_epochs": 3,
  "max_length": 2048,
  "warmup_ratio": 0.1,
  "checkpoint_steps": 1000,
  "output_dir": "./output",
  "device": "cuda",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 4,
  "use_wandb": true,
  "wandb_project": "qwen2.5-sft",
  "wandb_run_name": null
}
```

**不建议修改**:

* 减小 max_length → 截断长序列，影响学习能力
* 减小 batch_size 而不增加梯度累积 → 降低训练稳定性

**预期显存占用**（优化后约 19-24GB，相比 FP32 节省 50-60%）

---

## 5. 成功标准（可量化）

* **SC-001**: 用户能在 15 分钟内完成环境配置并启动训练。
* **SC-002**: 训练成功完成并生成可推理的模型检查点。
* **SC-003**: 代码简洁，30 分钟内新开发者可理解训练流程。
* **SC-004**: 依赖管理保证跨环境可重复性。

---

## 6. 假设与依赖

**假设**:

* 用户具备 Python 和深度学习基础
* GPU/CPU 资源充足
* 训练数据格式正确且质量良好
* 目标为微调预训练语言模型

**依赖**:

* 深度学习框架与库
* 训练数据可用且格式正确
* GPU 显存与存储空间
* uv 依赖管理工具可用

---
