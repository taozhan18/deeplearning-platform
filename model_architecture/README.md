# 模型架构 - Model Architecture

## 模块概述

模型架构模块是深度学习平台的核心组件，提供多种先进的神经网络架构实现，专为科学计算和工程应用设计。支持FNO、UNet、Transformer、Transolver、MeshGraphNet和MLP等6种主要模型类型，采用模块化设计，支持自动注册和扩展。

## 实际文件结构

```
model_architecture/
├── README.md                          # 本说明文档
└── src/
    ├── model_manager.py              # 模型管理核心类
    ├── model_template.py             # 模型模板基类
    ├── fno/                          # 傅里叶神经算子
    │   ├── fno.py                    # FNO模型实现
    │   ├── spectral_layers.py        # 谱层实现
    │   ├── README.md                 # FNO说明文档
    │   └── README_CN.md              # FNO中文说明
    ├── unet/                         # UNet架构
    │   ├── unet.py                   # UNet模型实现
    │   ├── README.md                 # UNet说明文档
    │   └── README_CN.md              # UNet中文说明
    ├── transformer/                  # Transformer模型
    │   ├── transformer.py            # Transformer实现
    │   ├── __init__.py               # 模块初始化
    │   ├── README.md                 # Transformer说明文档
    │   └── README_CN.md              # Transformer中文说明
    ├── transolver/                   # Transolver科学计算模型
    │   ├── transolver.py             # Transolver实现
    │   ├── Embedding.py              # 嵌入层
    │   ├── Physics_Attention.py      # 物理注意力机制
    │   ├── __init__.py               # 模块初始化
    │   ├── README.md                 # Transolver说明文档
    │   └── README_CN.md              # Transolver中文说明
    ├── meshgraphnet/                 # 图神经网络
    │   ├── meshgraphnet.py           # MeshGraphNet实现
    │   ├── __init__.py               # 模块初始化
    │   ├── README.md                 # MeshGraphNet说明文档
    │   └── README_CN.md              # MeshGraphNet中文说明
    └── mlp/                          # 多层感知机
        ├── mlp.py                    # MLP模型实现
        ├── README.md                 # MLP说明文档
        └── README_CN.md              # MLP中文说明
```

## 支持的模型架构

### 1. 傅里叶神经算子 (FNO)
**位置**: `src/fno/`

**适用场景**: 偏微分方程求解、物理场预测

**核心特性**:
- 基于傅里叶变换的谱方法
- 高效处理周期性边界条件
- 支持高维输入输出
- 参数高效的神经算子

**使用示例**:
```python
from src.model_manager import ModelManager

# 创建FNO模型
manager = ModelManager({})
model = manager.create_model(
    "fno",
    in_channels=3,
    out_channels=1,
    modes=12,
    width=32
)
```

### 2. UNet
**位置**: `src/unet/`

**适用场景**: 图像分割、图像到图像映射、物理场重建

**核心特性**:
- 经典的编码器-解码器结构
- 跳跃连接保留细节信息
- 支持多尺度特征提取
- 灵活的通道配置

**使用示例**:
```python
model = manager.create_model(
    "unet",
    in_channels=3,
    out_channels=1,
    features=[32, 64, 128, 256, 512]
)
```

### 3. Transformer
**位置**: `src/transformer/`

**适用场景**: 序列到序列映射、注意力机制应用

**核心特性**:
- 多头注意力机制
- 位置编码支持
- 可扩展的架构设计
- 支持变长输入

**使用示例**:
```python
model = manager.create_model(
    "transformer",
    d_model=128,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)
```

### 4. Transolver
**位置**: `src/transolver/`

**适用场景**: 科学计算专用、物理方程求解

**核心特性**:
- 物理信息神经网络
- 物理注意力机制
- 科学计算优化架构
- 物理约束集成

**使用示例**:
```python
model = manager.create_model(
    "transolver",
    d_model=256,
    n_layers=6,
    d_hidden=512
)
```

### 5. MeshGraphNet
**位置**: `src/meshgraphnet/`

**适用场景**: 网格数据处理、图结构学习

**核心特性**:
- 图神经网络架构
- 处理非结构化网格
- 支持动态图结构
- 网格自适应学习

**使用示例**:
```python
model = manager.create_model(
    "meshgraphnet",
    input_dim_node=3,
    input_dim_edge=4,
    hidden_dim=128,
    output_dim=1
)
```

### 6. MLP (多层感知机)
**位置**: `src/mlp/`

**适用场景**: 简单回归、分类任务、基线模型

**核心特性**:
- 经典全连接网络
- 灵活的网络深度
- 可调节的隐藏层大小
- 快速训练收敛

**使用示例**:
```python
model = manager.create_model(
    "mlp",
    input_dim=100,
    hidden_dims=[256, 128, 64],
    output_dim=1,
    activation="relu"
)
```

## 模型管理器 (ModelManager)

### 核心功能
**位置**: `src/model_manager.py`

**主要功能**:
- 自动模型注册和发现
- 模型参数验证和实例化
- 支持自定义模型扩展
- 模型元数据管理
- 统一接口调用

**使用示例**:
```python
from src.model_manager import ModelManager

# 创建管理器
manager = ModelManager({})

# 查看可用模型
print(manager.list_models())
# 输出: ['fno', 'unet', 'transformer', 'transolver', 'meshgraphnet', 'mlp']

# 创建模型
model = manager.create_model("fno", **model_params)

# 注册自定义模型
manager.register_custom_model("custom_model", CustomModelClass)
```

### 模型模板系统
**位置**: `src/model_template.py`

**功能**: 提供标准模型接口模板，确保所有模型实现一致性

## 模型配置参数

### FNO配置
```yaml
model:
  name: "fno"
  parameters:
    in_channels: 3      # 输入通道数
    out_channels: 1     # 输出通道数
    modes: 12          # 傅里叶模式数
    width: 32          # 隐藏层宽度
    n_layers: 4        # 网络层数
```

### UNet配置
```yaml
model:
  name: "unet"
  parameters:
    in_channels: 3      # 输入通道数
    out_channels: 1     # 输出通道数
    features: [32, 64, 128, 256, 512]  # 每层特征数
```

### Transformer配置
```yaml
model:
  name: "transformer"
  parameters:
    d_model: 128        # 模型维度
    nhead: 8           # 注意力头数
    num_encoder_layers: 6
    num_decoder_layers: 6
    dim_feedforward: 512
```

### Transolver配置
```yaml
model:
  name: "transolver"
  parameters:
    d_model: 256        # 模型维度
    n_layers: 6        # 层数
    d_hidden: 512      # 隐藏层维度
    n_head: 8          # 注意力头数
```

### MeshGraphNet配置
```yaml
model:
  name: "meshgraphnet"
  parameters:
    input_dim_node: 3   # 节点输入维度
    input_dim_edge: 4   # 边输入维度
    hidden_dim: 128     # 隐藏层维度
    output_dim: 1       # 输出维度
    n_layers: 5         # 网络层数
```

### MLP配置
```yaml
model:
  name: "mlp"
  parameters:
    input_dim: 100      # 输入维度
    hidden_dims: [256, 128, 64]  # 隐藏层维度列表
    output_dim: 1       # 输出维度
    activation: "relu"  # 激活函数
    dropout: 0.1        # dropout比例
```

## 自定义模型扩展

### 创建自定义模型
```python
# 创建新模型文件 my_model.py
import torch.nn as nn
from src.model_template import BaseModel

class MyModel(BaseModel):
    HYPERPARAMETERS = {
        'input_dim': int,
        'hidden_dim': int,
        'output_dim': int
    }
    
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
from src.model_manager import ModelManager
manager = ModelManager({})
manager.register_custom_model("mymodel", MyModel)

# 使用自定义模型
model = manager.create_model("mymodel", input_dim=100, hidden_dim=64, output_dim=1)
```

## 测试验证

### 测试方法
模型架构的测试位于上级目录的 `test/model/` 中：
- `test/model/test_fno.py` - FNO模型测试
- `test/model/test_unet.py` - UNet模型测试
- `test/model/test_transformer.py` - Transformer模型测试
- `test/model/test_transolver.py` - Transolver模型测试
- `test/model/test_meshgraphnet.py` - MeshGraphNet模型测试
- `test/model/test_mlp.py` - MLP模型测试
- `test/model/test_platform.py` - 平台通用功能测试

### 运行测试
```bash
# 运行所有模型测试
python -m pytest test/model/ -v

# 运行特定模型测试
python -m pytest test/model/test_fno.py -v

# 运行模型管理器测试
python -m pytest test/model/test_platform.py -v
```

## 使用示例

### 1. 基础模型创建
```python
from src.model_manager import ModelManager

# 初始化管理器
manager = ModelManager({})

# 创建FNO模型用于PDE求解
fno_model = manager.create_model(
    "fno",
    in_channels=3,      # [u, v, p] 速度场和压力
    out_channels=1,     # 预测的压力场
    modes=12,
    width=32
)

# 创建UNet用于图像重建
unet_model = manager.create_model(
    "unet",
    in_channels=1,      # 输入图像
    out_channels=1,     # 重建图像
    features=[64, 128, 256, 512, 1024]
)

# 创建MeshGraphNet用于网格学习
mesh_model = manager.create_model(
    "meshgraphnet",
    input_dim_node=3,   # 节点特征
    input_dim_edge=4,   # 边特征
    hidden_dim=128,
    output_dim=1        # 预测值
)
```

### 2. 多模型比较
```python
# 定义模型配置
models_config = {
    "fno": {
        "in_channels": 3,
        "out_channels": 1,
        "modes": 12,
        "width": 32
    },
    "unet": {
        "in_channels": 3,
        "out_channels": 1,
        "features": [32, 64, 128, 256, 512]
    },
    "transolver": {
        "d_model": 256,
        "n_layers": 6,
        "d_hidden": 512
    }
}

# 批量创建和测试模型
models = {}
for name, params in models_config.items():
    models[name] = manager.create_model(name, **params)
    print(f"{name} 模型参数量: {sum(p.numel() for p in models[name].parameters())}")
```

### 3. 模型架构分析
```python
# 检查模型结构
for model_name, model in models.items():
    print(f"\n{model_name.upper()} Architecture:")
    print(model)
    print(f"总参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
```

## 开发计划

### 短期计划（1个月内）
1. **架构优化**
   - 添加残差连接选项
   - 实现注意力机制增强
   - 支持深度可分离卷积

2. **配置增强**
   - 添加模型配置文件模板
   - 实现自动超参数搜索
   - 支持模型架构可视化

3. **扩展支持**
   - 添加ResNet架构
   - 实现Vision Transformer
   - 支持EfficientNet

### 中长期计划（3-6个月）
1. **高级功能**
   - 添加神经架构搜索(NAS)
   - 实现模型压缩技术
   - 支持知识蒸馏

2. **性能优化**
   - 集成模型量化
   - 添加剪枝算法
   - 支持动态网络架构

3. **科研支持**
   - 添加物理信息神经网络
   - 实现符号回归网络
   - 支持可解释性架构

## 故障排除

### 常见问题
1. **内存不足**
   - 减小模型宽度或深度
   - 使用更高效的架构
   - 启用梯度检查点

2. **模型不匹配**
   - 验证输入输出维度
   - 检查数据格式一致性
   - 确认任务类型匹配

3. **训练问题**
   - 调整模型复杂度
   - 验证初始化方法
   - 检查激活函数选择

### 调试工具
```python
# 检查模型信息
from src.model_manager import ModelManager

manager = ModelManager({})
model = manager.create_model("fno", **params)

# 打印模型摘要
print(manager.get_model_info("fno"))

# 验证参数
print(manager.validate_parameters("fno", params))

# 检查兼容性
print(manager.check_model_data_compatibility("fno", input_shape, output_shape))
```

## 最佳实践

### 1. 模型选择
- 物理问题优先选择FNO或Transolver
- 图像任务选择UNet或Transformer
- 图数据选择MeshGraphNet
- 简单任务使用MLP作为基线

### 2. 架构设计
- 从小模型开始逐步扩展
- 使用交叉验证选择最佳架构
- 考虑计算资源限制
- 平衡模型复杂度和性能

### 3. 实验管理
- 为每个模型创建独立配置文件
- 记录模型架构和超参数
- 使用版本控制管理模型文件
- 建立模型性能基准

### 4. 扩展开发
- 遵循BaseModel接口规范
- 添加完整的HYPERPARAMETERS定义
- 提供详细的使用文档
- 包含充分的测试用例

---

*最后更新：2025年9月1日*