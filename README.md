# Low-Code Deep Learning Platform
低代码深度学习平台

## 项目概述 (Project Overview)

这是一个专为科学计算和工程应用设计的低代码深度学习平台，支持多种先进神经网络架构，包括FNO、UNet、Transformer、Transolver和MeshGraphNet等模型。平台采用模块化设计，通过配置文件驱动整个训练和推理流程。

This is a low-code deep learning platform designed specifically for scientific computing and engineering applications, supporting advanced neural network architectures including FNO, UNet, Transformer, Transolver, and MeshGraphNet models. The platform features a modular design driven entirely by configuration files.

## 核心特性 (Core Features)

### 🧠 支持的模型架构 (Supported Model Architectures)
- **FNO (Fourier Neural Operator)** - 用于PDE求解的傅里叶神经算子
- **UNet** - 经典的图像分割和回归网络
- **Transformer** - 注意力机制模型
- **Transolver** - 科学计算专用Transformer变体
- **MeshGraphNet** - 基于图神经网络的网格处理模型
- **MLP** - 多层感知机

### 📊 数据处理系统 (Data Processing System)
- **多源数据支持** - 支持多个输入数据源的并行处理
- **多种数据格式** - CSV、JSON、NPY、NPZ格式支持
- **智能数据标准化** - 支持Standard、MinMax、Robust标准化方法
- **动态数据验证** - 运行时数据一致性检查

### ⚙️ 训练引擎 (Training Engine)
- **多优化器支持** - Adam、SGD、AdamW等主流优化器
- **丰富损失函数** - CrossEntropy、MSE、L1、BCE等多种损失函数
- **学习率调度** - Step、Exponential、CosineAnnealing调度器
- **自定义预处理** - 支持Python文件自定义数据预处理函数
- **GPU/CPU自动适配** - 智能设备选择和内存管理

## 快速开始 (Quick Start)

### 环境要求 (Requirements)
```bash
Python 3.8+
PyTorch 1.9+
numpy
pandas
scikit-learn
pyyaml
```

### 安装步骤 (Installation)

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd deeplearning-platform
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行训练**
   ```bash
   python main/train.py --config configs/example_config.yaml
   ```

### 配置文件示例 (Configuration Example)

创建配置文件 `configs/my_training.yaml`:

```yaml
# 数据配置 (Data Configuration)
data:
  train_features_path: "data/train_X.csv"
  train_targets_path: "data/train_y.csv"
  test_features_path: "data/test_X.csv"
  test_targets_path: "data/test_y.csv"
  batch_size: 64
  shuffle: true
  normalize: true
  normalization_method: "standard"

# 模型配置 (Model Configuration)
model:
  name: "fno"
  parameters:
    in_channels: 3
    out_channels: 1
    modes: 12
    width: 32

# 训练配置 (Training Configuration)
training:
  epochs: 100
  device: "cuda"  # or "cpu"

# 优化器配置 (Optimizer Configuration)
optimizer:
  name: "adam"
  parameters:
    lr: 0.001
    weight_decay: 1e-4

# 损失函数配置 (Criterion Configuration)
criterion:
  name: "mse"

# 学习率调度器配置 (Scheduler Configuration)
scheduler:
  name: "step"
  parameters:
    step_size: 30
    gamma: 0.1

# 输出配置 (Output Configuration)
output:
  model_path: "models/trained_model.pth"
  history_path: "results/training_history.json"
```

## 多源数据处理 (Multi-Source Data Processing)

平台支持同时处理多个输入数据源，适用于复杂的多模态学习场景：

```yaml
data:
  train_features_paths:
    pressure: "data/train_pressure.csv"
    velocity: "data/train_velocity.csv"
    temperature: "data/train_temp.csv"
  train_targets_path: "data/train_solution.csv"
  test_features_paths:
    pressure: "data/test_pressure.csv"
    velocity: "data/test_velocity.csv"
    temperature: "data/test_temp.csv"
  test_targets_path: "data/test_solution.csv"
```

## 项目结构 (Project Structure)

```
deeplearning-platform/
├── main/
│   └── train.py                 # 主训练脚本
├── data_loader/
│   └── src/
│       └── data_loader.py       # 数据加载和处理模块
├── model_architecture/
│   └── src/
│       └── model_manager.py     # 模型管理和注册系统
├── training_engine/
│   └── src/
│       └── training_engine.py   # 训练引擎核心
├── configs/                     # 配置文件目录
├── data/                        # 数据文件目录
├── models/                      # 训练好的模型保存目录
├── results/                     # 训练结果和日志目录
└── requirements.txt             # Python依赖列表
```

## 模块详解 (Module Details)

### 1. 数据加载模块 (Data Loader Module)

**位置**: `data_loader/src/data_loader.py`

**核心功能**:
- `DataLoaderModule`: 主数据加载类，支持单源和多源数据处理
- `DataNormalizer`: 数据标准化工具，支持多种标准化方法
- `MultiSourceDataset`: 多源数据集类，支持复杂输入结构
- `BaseDataset`: 基础数据集类，处理标准数据格式

**使用示例**:
```python
from data_loader import DataLoaderModule

config = {
    "train_features_path": "data/train.csv",
    "test_features_path": "data/test.csv",
    "batch_size": 32,
    "normalize": True
}

data_loader = DataLoaderModule(config)
data_loader.prepare_datasets()
data_loader.create_data_loaders()
train_loader, test_loader = data_loader.get_data_loaders()
```

### 2. 模型管理模块 (Model Manager)

**位置**: `model_architecture/src/model_manager.py`

**核心功能**:
- 自动模型注册和发现
- 模型参数验证和实例化
- 支持自定义模型扩展
- 模型元数据管理

**使用示例**:
```python
from model_manager import ModelManager

manager = ModelManager({})
model = manager.create_model("fno", in_channels=3, out_channels=1, modes=12)
print(manager.list_models())  # 查看所有可用模型
```

### 3. 训练引擎 (Training Engine)

**位置**: `training_engine/src/training_engine.py`

**核心功能**:
- 完整的训练循环管理
- 自动设备选择和内存优化
- 训练和验证指标跟踪
- 模型保存和加载
- 自定义预处理函数支持

**使用示例**:
```python
from training_engine import TrainingEngine

config = {"epochs": 100, "device": "cuda"}
trainer = TrainingEngine(config)
trainer.set_model(model)
trainer.configure_optimizer("adam", lr=0.001)
trainer.configure_criterion("mse")
history = trainer.train(train_loader, test_loader)
```

## 高级用法 (Advanced Usage)

### 自定义模型注册 (Custom Model Registration)

```python
from model_manager import ModelManager

class MyCustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# 注册自定义模型
manager = ModelManager({})
manager.register_custom_model("mycustom", MyCustomModel)
model = manager.create_model("mycustom", input_dim=10, hidden_dim=64, output_dim=1)
```

### 自定义预处理函数 (Custom Preprocessing)

创建 `my_preprocess.py`:
```python
def preprocess_fn(data):
    """自定义数据预处理函数"""
    if isinstance(data, dict):
        # 处理多源数据
        combined = torch.cat([data[key] for key in sorted(data.keys())], dim=1)
        return combined
    return data
```

在配置文件中使用:
```yaml
training:
  preprocess_fn: "my_preprocess.py"
```

### 批量实验配置 (Batch Experiment Configuration)

使用JSON/YAML配置文件管理多个实验:

```bash
# 运行多个配置实验
for config in configs/experiments/*.yaml; do
    python main/train.py --config "$config"
done
```

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

1. **CUDA内存不足**
   ```yaml
   training:
     device: "cpu"  # 切换到CPU
     batch_size: 16  # 减小批次大小
   ```

2. **模型导入错误**
   - 确保所有模型文件在正确的目录结构下
   - 检查模型类的超参数定义

3. **数据格式不匹配**
   - 使用`data_loader.get_dataset_info()`检查数据信息
   - 确认输入输出维度与模型配置匹配

### 性能优化建议 (Performance Optimization)

1. **数据加载优化**
   - 使用`.npy`格式替代CSV以提高加载速度
   - 启用数据标准化缓存

2. **训练加速**
   - 使用GPU训练 (`device: "cuda"`)
   - 调整批次大小以平衡内存和速度
   - 使用混合精度训练（需要代码扩展）

3. **内存管理**
   - 监控GPU内存使用
   - 使用梯度累积处理大批次

## 扩展指南 (Extension Guide)

### 添加新模型架构 (Adding New Model Architectures)

1. 在对应目录创建模型文件（如`fno/fno.py`）
2. 定义模型类和HYPERPARAMETERS静态变量
3. 模型会自动被ModelManager注册

### 添加新数据格式支持 (Adding New Data Format Support)

在`DataLoaderModule.load_data()`方法中添加新的文件格式处理逻辑。

### 添加新优化器或损失函数 (Adding New Optimizers or Loss Functions)

在`TrainingEngine`类中添加新的优化器和损失函数配置。

## 许可证 (License)

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情

## 贡献指南 (Contributing)

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 支持与联系 (Support & Contact)

- 📧 问题反馈：提交GitHub Issue
- 📖 文档更新：欢迎贡献文档改进
- 💡 功能建议：通过Issue或Discussion提出

---

*最后更新：2025年9月1日*