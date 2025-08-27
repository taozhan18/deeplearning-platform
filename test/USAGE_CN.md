# 使用主训练函数处理不同任务

本文档说明如何使用低代码平台中的主训练函数处理各种深度学习任务。

## 概述

[main/train.py](file:///home/zt/workspace/deeplearning-platform/main/train.py) 中的主训练函数设计为通用的，可以通过简单地更改配置文件来处理各种深度学习任务。这种方法避免了代码重复，并确保了不同任务之间的一致性。

## 使用方法

1. 为特定任务创建配置文件
2. 使用配置文件运行主训练函数：

```bash
python main/train.py --config your_config.yaml
```

## 配置文件结构

所有配置文件都应遵循以下结构：

```yaml
# 数据配置
data:
  # 数据路径和参数
  train_features_path: "path/to/train_features"
  train_targets_path: "path/to/train_targets"
  test_features_path: "path/to/test_features"
  test_targets_path: "path/to/test_targets"
  batch_size: 32
  shuffle: true

# 模型配置
model:
  name: "model_name"
  parameters:
    # 模型特定参数

# 训练配置
training:
  epochs: 10
  device: "cpu"  # 或 "cuda"

# 优化器配置
optimizer:
  name: "adam"
  parameters:
    lr: 0.001

# 损失函数配置
criterion:
  name: "cross_entropy"  # 或用于回归任务的 "mse"

# 调度器配置（可选）
scheduler:
  name: "step"
  parameters:
    step_size: 5
    gamma: 0.1

# 输出配置
output:
  model_path: "path/to/save/model.pth"
  history_path: "path/to/save/history.json"
```

## 示例

### 1. 标准分类任务

使用模板模型的标准分类任务：

```yaml
model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2
```

### 2. FNO1D任务

使用1D数据的FNO1D任务：

```yaml
data:
  # FNO1D特定数据配置
  train_features_path: "data/fno1d/train_features.npy"
  train_targets_path: "data/fno1d/train_targets.npy"
  test_features_path: "data/fno1d/test_features.npy"
  test_targets_path: "data/fno1d/test_targets.npy"
  batch_size: 16
  shuffle: true
  fno1d_data: true

model:
  name: "fno"
  parameters:
    in_channels: 1
    out_channels: 1
    dimension: 1
    # 其他FNO特定参数
```

## 优势

1. **代码复用性**：无需为每个任务编写单独的训练脚本
2. **一致性**：所有任务遵循相同的训练过程
3. **可维护性**：对训练过程的更改只需在一个地方进行
4. **灵活性**：通过创建新的配置文件可以轻松添加新模型和任务

## 添加新任务

要添加新任务：

1. 创建新的配置文件，遵循上述结构
2. 确保数据采用支持的格式（CSV、JSON、NPY、NPZ）
3. 如有必要，扩展数据加载器以支持新的数据格式
4. 如有必要，在模型管理器中注册新模型
5. 使用新配置文件运行训练