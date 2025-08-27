# 低代码深度学习训练平台

一个低代码深度学习训练平台，让用户专注于模型和数据，而不是实现细节。

## 功能特点

- **模块化设计**：将数据加载、模型架构和训练引擎分离为独立模块
- **预定义模型**：内置来自PyTorch的流行模型架构
- **简单配置**：指定数据集路径并选择模型架构即可开始训练
- **灵活使用**：支持编程和UI操作方式
- **可扩展性**：易于添加新模型和功能

## 模块介绍

1. [数据加载](./data_loader) - 处理数据集加载和预处理
2. [模型架构](./model_architecture) - 提供预定义的模型架构和模板
3. [训练引擎](./training_engine) - 控制训练过程

## 安装

```bash
pip install -r requirements.txt
```

## 可用模型

有关可用模型及其参数的完整列表，请参见：
- [Model Guide](MODEL_GUIDE.md) - 所有可用模型的详细信息
- [模型指南](MODEL_GUIDE_CN.md) - 所有可用模型的详细信息（中文版）

## 使用方法

用户只需：
1. 指定训练和测试数据集的路径
2. 选择模型架构
3. 配置训练超参数（可选）
4. 开始训练

平台将自动处理整个训练工作流程。

### 配置驱动方式

创建一个配置文件（YAML或JSON格式），指定以下内容：

```yaml
# 示例 config.yaml
data:
  train_features_path: "data/train_features.csv"
  train_targets_path: "data/train_targets.csv"
  test_features_path: "data/test_features.csv"
  test_targets_path: "data/test_targets.csv"
  batch_size: 32

model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2

training:
  epochs: 10
  device: "cpu"

optimizer:
  name: "adam"
  parameters:
    lr: 0.001

criterion:
  name: "cross_entropy"
```

然后运行训练：

```bash
python main/train.py --config config.yaml
```

### 编程方式

```python
from data_loader.src.data_loader import DataLoaderModule
from model_architecture.src.model_manager import ModelManager
from training_engine.src.training_engine import TrainingEngine

# 配置数据加载
data_config = {
    'train_features_path': 'data/train_features.csv',
    'train_targets_path': 'data/train_targets.csv',
    'batch_size': 32
}

# 加载数据
data_loader = DataLoaderModule(data_config)
data_loader.prepare_datasets()
data_loader.create_data_loaders()
train_loader, test_loader = data_loader.get_data_loaders()

# 配置模型
model_manager = ModelManager({})
model = model_manager.create_model('ModelTemplate', 
                                   input_size=784, 
                                   hidden_size=256, 
                                   num_classes=10)

# 配置训练
training_config = {'epochs': 10, 'device': 'cpu'}
training_engine = TrainingEngine(training_config)
training_engine.set_model(model)
training_engine.configure_optimizer('adam', lr=0.001)
training_engine.configure_criterion('cross_entropy')

# 开始训练
history = training_engine.train(train_loader, test_loader)
```

## 项目结构

```
deeplearning-platform/
├── data_loader/
│   ├── src/
│   │   └── data_loader.py
│   ├── README.md
│   └── README_CN.md
├── model_architecture/
│   ├── src/
│   │   └── model_manager.py
│   ├── templates/
│   │   ├── model_template.py
│   │   ├── README.md
│   │   └── README_CN.md
│   ├── README.md
│   └── README_CN.md
├── training_engine/
│   ├── src/
│   │   └── training_engine.py
│   ├── README.md
│   └── README_CN.md
├── main/
│   ├── train.py
│   └── config.yaml
├── requirements.txt
├── README.md
└── README_CN.md
```