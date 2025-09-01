# 训练引擎 - Training Engine

## 模块概述

训练引擎模块是整个深度学习平台的核心训练组件，提供完整的训练循环管理、优化器配置、损失函数设置和学习率调度等功能。专为科学计算和工程应用优化，支持GPU/CPU自动适配和自定义预处理。

## 实际文件结构

```
training_engine/
├── README.md                          # 本说明文档
└── src/
    ├── training_engine.py            # 训练引擎核心实现
    └── __init__.py                   # 模块初始化文件
```

## 核心功能

### 1. 训练引擎类 (TrainingEngine)
**位置**: `src/training_engine.py`

**主要功能**:
- 完整的训练循环管理
- 多优化器支持（Adam, SGD, AdamW）
- 丰富损失函数（MSE, CrossEntropy, L1, BCE等）
- 学习率调度器（Step, Exponential, CosineAnnealing）
- GPU/CPU自动设备选择
- 模型保存和加载
- 自定义预处理函数支持
- 训练和验证指标跟踪
- 内存优化和梯度累积

**使用示例**:
```python
from src.training_engine import TrainingEngine

# 基础配置
config = {
    "epochs": 100,
    "device": "cuda",
    "save_every": 10,
    "early_stopping": True
}

# 创建训练引擎
trainer = TrainingEngine(config)
trainer.set_model(model)
trainer.configure_optimizer("adam", lr=0.001, weight_decay=1e-4)
trainer.configure_criterion("mse")
trainer.configure_scheduler("step", step_size=30, gamma=0.1)

# 开始训练
history = trainer.train(train_loader, test_loader)
```

### 2. 优化器配置
**支持的优化器**:
- **Adam**: 自适应矩估计优化器
- **SGD**: 随机梯度下降
- **AdamW**: 权重衰减Adam

**配置示例**:
```python
# Adam优化器
trainer.configure_optimizer("adam", lr=0.001, betas=(0.9, 0.999))

# SGD优化器
trainer.configure_optimizer("sgd", lr=0.01, momentum=0.9)

# AdamW优化器
trainer.configure_optimizer("adamw", lr=0.001, weight_decay=0.01)
```

### 3. 损失函数
**支持的损失函数**:
- **MSELoss**: 均方误差损失
- **CrossEntropyLoss**: 交叉熵损失
- **L1Loss**: L1损失
- **BCELoss**: 二元交叉熵损失
- **BCEWithLogitsLoss**: 带logits的二元交叉熵

**配置示例**:
```python
# MSE损失
trainer.configure_criterion("mse")

# 交叉熵损失
trainer.configure_criterion("crossentropy")

# L1损失
trainer.configure_criterion("l1")
```

### 4. 学习率调度器
**支持的调度器**:
- **StepLR**: 步长调度器
- **ExponentialLR**: 指数衰减
- **CosineAnnealingLR**: 余弦退火
- **ReduceLROnPlateau**: 基于验证损失的自适应调度

**配置示例**:
```python
# Step调度器
trainer.configure_scheduler("step", step_size=30, gamma=0.1)

# 指数调度器
trainer.configure_scheduler("exponential", gamma=0.95)

# 余弦退火
trainer.configure_scheduler("cosine", T_max=50)
```

### 5. 自定义预处理
**功能**: 支持外部Python文件定义自定义预处理函数

**使用示例**:
```python
# 创建自定义预处理文件 my_preprocess.py
def preprocess_fn(data):
    """自定义数据预处理"""
    if isinstance(data, dict):
        # 处理多源数据
        combined = torch.cat([data[key] for key in sorted(data.keys())], dim=1)
        return combined
    return data

# 在训练引擎中使用
trainer.set_custom_preprocess("my_preprocess.py")
```

## 高级功能

### 1. 模型保存和加载
```python
# 设置保存配置
config = {
    "model_path": "models/best_model.pth",
    "save_every": 10,
    "save_best_only": True
}

# 自动保存最佳模型
trainer = TrainingEngine(config)

# 手动保存模型
trainer.save_model("models/manual_save.pth")

# 加载模型
trainer.load_model("models/best_model.pth")
```

### 2. 训练监控
```python
# 获取训练历史
history = trainer.get_training_history()
print(f"训练损失: {history['train_loss']}")
print(f"验证损失: {history['val_loss']}")
print(f"学习率: {history['learning_rate']}")
```

### 3. 内存优化
```python
# 启用梯度累积
trainer.enable_gradient_accumulation(accumulation_steps=4)

# 内存清理
trainer.clear_memory()
```

### 4. 早停机制
```python
# 配置早停
config = {
    "early_stopping": True,
    "patience": 10,
    "min_delta": 0.001
}
```

## 配置参数详解

### 主要配置项
```yaml
training:
  # 基本训练参数
  epochs: 100
  device: "cuda"  # "cuda" 或 "cpu"
  
  # 模型保存
  model_path: "models/trained_model.pth"
  save_every: 10
  save_best_only: true
  
  # 早停设置
  early_stopping: true
  patience: 10
  min_delta: 0.001
  
  # 梯度累积
  gradient_accumulation_steps: 1
  
  # 日志设置
  log_every: 10
  verbose: true

optimizer:
  name: "adam"
  parameters:
    lr: 0.001
    weight_decay: 1e-4
    betas: [0.9, 0.999]

criterion:
  name: "mse"

scheduler:
  name: "step"
  parameters:
    step_size: 30
    gamma: 0.1
    
# 自定义预处理
preprocess_fn: "custom_preprocess.py"
```

## 测试验证

### 测试方法
训练引擎的测试位于上级目录的 `test/training_engine/` 中：
- `test/training_engine/test_training_engine.py` - 训练引擎核心测试
- `test/training_engine/test_custom_preprocessing.py` - 自定义预处理测试

### 运行测试
```bash
# 运行训练引擎测试
python -m pytest test/training_engine/ -v

# 运行特定测试
python -m pytest test/training_engine/test_training_engine.py::test_optimizer_config -v
```

## 使用示例

### 1. 完整训练流程
```python
from training_engine.src.training_engine import TrainingEngine

# 配置训练
config = {
    "epochs": 50,
    "device": "cuda",
    "model_path": "models/fno_model.pth",
    "save_best_only": True,
    "early_stopping": True,
    "patience": 15
}

# 创建训练器
trainer = TrainingEngine(config)

# 设置模型和数据
trainer.set_model(model)
trainer.set_data_loaders(train_loader, test_loader)

# 配置训练组件
trainer.configure_optimizer("adam", lr=0.001, weight_decay=1e-4)
trainer.configure_criterion("mse")
trainer.configure_scheduler("cosine", T_max=50)

# 开始训练
history = trainer.train()

# 保存训练历史
import json
with open("results/training_history.json", "w") as f:
    json.dump(history, f)
```

### 2. 恢复训练
```python
# 从检查点恢复
trainer.load_checkpoint("checkpoints/epoch_25.pth")
trainer.train(resume_from=26)
```

### 3. 多GPU训练
```python
# 自动多GPU支持
config = {
    "device": "cuda",
    "multi_gpu": True,
    "distributed": False
}
```

## 开发计划

### 短期计划（1个月内）
1. **功能增强**
   - 添加混合精度训练支持
   - 实现梯度裁剪功能
   - 添加学习率热身

2. **监控改进**
   - 集成Weights & Biases日志
   - 添加TensorBoard支持
   - 实现实时训练可视化

3. **检查点系统**
   - 添加训练中断恢复
   - 实现增量保存
   - 支持断点续训

### 中长期计划（3-6个月）
1. **高级功能**
   - 支持分布式训练
   - 添加自动超参数调优
   - 实现模型剪枝

2. **性能优化**
   - 集成NVIDIA Apex加速
   - 添加动态批大小调整
   - 支持异步数据加载

3. **实验管理**
   - 添加实验跟踪系统
   - 实现模型版本管理
   - 支持A/B测试框架

## 故障排除

### 常见问题
1. **CUDA内存不足**
   - 减小批次大小
   - 启用梯度检查点
   - 使用CPU训练

2. **训练速度慢**
   - 检查数据加载瓶颈
   - 优化模型结构
   - 使用更高效的优化器

3. **模型不收敛**
   - 调整学习率
   - 检查数据标准化
   - 验证损失函数选择

### 调试工具
```python
# 检查训练配置
print(trainer.get_config_summary())

# 验证数据流
trainer.validate_data_loaders()

# 检查设备信息
print(trainer.get_device_info())

# 内存使用监控
trainer.enable_memory_profiling()
```

## 最佳实践

### 1. 训练准备
- 使用GPU训练获得最佳性能
- 合理设置批次大小平衡速度和内存
- 启用早停避免过拟合

### 2. 超参数调优
- 使用学习率调度器
- 实验不同的优化器组合
- 监控验证损失调整训练

### 3. 模型保存
- 定期保存检查点
- 保存最佳验证损失模型
- 保持训练历史记录

### 4. 调试技巧
- 先在小数据集上验证
- 使用简单的模型结构测试
- 逐步增加复杂度

---

*最后更新：2025年9月1日*