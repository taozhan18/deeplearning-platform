# 数据加载模块 - Data Loader Module

## 模块概述

数据加载模块提供灵活的数据处理功能，支持多种数据格式和多源数据输入。基于实际代码实现，专注于科学计算和工程应用的数据需求。

## 实际文件结构

```
data_loader/
├── README.md                          # 本说明文档
├── moose/
│   └── moose_ml_automation.py        # MOOSE框架机器学习自动化脚本
└── src/
    ├── data_loader.py                # 主要数据加载类实现
    └── __init__.py                   # 模块初始化文件
```

## 核心功能

### 1. 数据加载类 (DataLoaderModule)
**位置**: `src/data_loader.py`

**主要功能**:
- 支持CSV、JSON、NPY、NPZ格式数据加载
- 单源和多源数据处理
- 数据标准化（StandardScaler, MinMaxScaler, RobustScaler）
- PyTorch DataLoader集成
- 内存映射加载大文件支持

**使用示例**:
```python
from src.data_loader import DataLoaderModule

# 单源数据配置
config = {
    "train_features_path": "data/train_X.npy",
    "train_targets_path": "data/train_y.npy",
    "batch_size": 32,
    "normalize": True,
    "normalization_method": "standard"
}

loader = DataLoaderModule(config)
loader.prepare_datasets()
train_loader, test_loader = loader.get_data_loaders()
```

### 2. 数据标准化 (DataNormalizer)
**位置**: `src/data_loader.py`

**标准化方法**:
- **StandardScaler**: 均值为0，标准差为1
- **MinMaxScaler**: 缩放到[0,1]区间
- **RobustScaler**: 使用中位数和四分位数

**使用示例**:
```python
from src.data_loader import DataNormalizer

normalizer = DataNormalizer(method="standard")
data = {"features": np.random.randn(100, 10)}
normalized = normalizer.fit_transform(data)
```

### 3. 多源数据集 (MultiSourceDataset)
**位置**: `src/data_loader.py`

**功能**: 处理多个输入数据源
```python
# 多源数据配置
config = {
    "train_features_paths": {
        "pressure": "data/pressure.npy",
        "velocity": "data/velocity.npy"
    },
    "train_targets_path": "data/targets.npy",
    "batch_size": 16
}

loader = DataLoaderModule(config)
dataset_info = loader.get_dataset_info()
print(f"数据源: {dataset_info['train_data_sources']}")
```

### 4. MOOSE框架集成
**位置**: `moose/moose_ml_automation.py`

**功能**: 自动化MOOSE仿真与机器学习集成
- 自动处理MOOSE仿真数据
- 支持参数-结果映射
- 集成机器学习模型训练

## 支持的数据格式

### 1. CSV格式
```python
# 无表头CSV
1.0,2.0,3.0
4.0,5.0,6.0

# 有表头CSV
feature1,feature2,feature3
1.0,2.0,3.0
```

### 2. JSON格式
```json
[
  {"feature1": 1.0, "feature2": 2.0},
  {"feature1": 3.0, "feature2": 4.0}
]
```

### 3. NumPy格式
- **NPY**: 单个数组
- **NPZ**: 多个压缩数组

## 使用示例

### 1. 基础数据加载
```python
import numpy as np
from src.data_loader import DataLoaderModule

# 创建测试数据
X = np.random.randn(100, 64, 64)
y = np.random.randn(100, 64, 64)
np.save("data/train_X.npy", X)
np.save("data/train_y.npy", y)

# 配置加载器
config = {
    "train_features_path": "data/train_X.npy",
    "train_targets_path": "data/train_y.npy",
    "batch_size": 32,
    "shuffle": True,
    "normalize": True
}

loader = DataLoaderModule(config)
loader.prepare_datasets()
loader.create_data_loaders()
train_loader, test_loader = loader.get_data_loaders()
```

### 2. 多源数据处理
```python
# 创建多源数据
pressure = np.random.randn(100, 32, 32)
velocity = np.random.randn(100, 32, 32, 2)
np.save("data/pressure.npy", pressure)
np.save("data/velocity.npy", velocity)

config = {
    "train_features_paths": {
        "pressure": "data/pressure.npy",
        "velocity": "data/velocity.npy"
    },
    "train_targets_path": "data/targets.npy",
    "batch_size": 16
}

loader = DataLoaderModule(config)
```

### 3. MOOSE数据自动化
```python
# 使用MOOSE自动化脚本
from moose.moose_ml_automation import MooseMLAutomation

automation = MooseMLAutomation()
automation.process_simulation_data("moose_simulations/")
automation.prepare_ml_dataset()
```

## 配置参数详解

### 主要配置项
```yaml
data:
  # 单源数据 (Single Source Data)
  train_features_path: "path/to/features"     # 训练特征数据路径
  train_targets_path: "path/to/targets"       # 训练目标数据路径
  test_features_path: "path/to/features"      # 测试特征数据路径
  test_targets_path: "path/to/targets"        # 测试目标数据路径
  
  # 多源数据 (Multi-source Data)
  train_features_paths:                       # 多源训练特征数据路径字典
    source1: "path/to/source1"
    source2: "path/to/source2"
  test_features_paths:                        # 多源测试特征数据路径字典
    source1: "path/to/source1"
    source2: "path/to/source2"
  
  # 数据加载参数 (Data Loading Parameters)
  batch_size: 32                             # 批次大小，默认值: 32
  shuffle: true                              # 是否打乱训练数据，默认值: true
  
  # 标准化设置 (Normalization Settings)
  normalize: true                            # 是否标准化特征数据，默认值: false
  normalization_method: "standard"           # 标准化方法，可选: standard, minmax, robust，默认值: standard
  normalize_targets: false                   # 是否标准化目标数据，默认值: false
  
  # 内存优化 (Memory Optimization)
  memory_map: true                           # 是否使用内存映射加载大文件，默认值: false
```

### 配置参数详细说明

| 参数名 | 类型 | 必需 | 默认值 | 描述 |
|-------|------|------|--------|------|
| train_features_path | str | 单源模式必需 | 无 | 训练特征数据文件路径 |
| train_targets_path | str | 可选 | None | 训练目标数据文件路径 |
| test_features_path | str | 单源模式必需 | 无 | 测试特征数据文件路径 |
| test_targets_path | str | 可选 | None | 测试目标数据文件路径 |
| train_features_paths | dict | 多源模式必需 | {} | 多源训练特征数据路径字典 |
| test_features_paths | dict | 多源模式必需 | {} | 多源测试特征数据路径字典 |
| batch_size | int | 可选 | 32 | 批次大小 |
| shuffle | bool | 可选 | true | 是否打乱训练数据 |
| normalize | bool | 可选 | false | 是否标准化特征数据 |
| normalization_method | str | 可选 | "standard" | 标准化方法 ("standard", "minmax", "robust") |
| normalize_targets | bool | 可选 | false | 是否标准化目标数据 |
| memory_map | bool | 可选 | false | 是否使用内存映射加载大文件 |

## 测试验证

### 测试方法
数据加载模块的测试位于上级目录的 `test/data_loader/` 中：
- `test/data_loader/test_data_loader.py` - 数据加载核心测试
- `test/data_loader/integration_test.py` - 数据集成测试
- `test/data_loader/run_tests.py` - 数据模块测试运行器

### 运行测试
```bash
# 运行数据加载测试
cd test/data_loader
python run_tests.py
# 或
python -m pytest test/data_loader/ -v
```

## 开发计划

### 短期计划（1个月内）
1. **数据格式扩展**
   - 支持Parquet格式
   - 添加HDF5支持

2. **性能优化**
   - 添加数据缓存机制
   - 支持内存映射加载大文件

3. **验证增强**
   - 添加数据一致性检查
   - 增加异常值检测

### 中长期计划（3-6个月）
1. **高级功能**
   - 支持流式数据加载
   - 添加数据增强功能
   - 集成数据可视化工具

2. **性能监控**
   - 添加数据加载性能统计
   - 内存使用分析

3. **MOOSE集成增强**
   - 扩展MOOSE数据格式支持
   - 添加MOOSE仿真参数优化

## 故障排除

### 常见问题
1. **内存不足**
   - 减小批次大小
   - 使用数据分块加载
   - 启用内存映射

2. **数据格式错误**
   - 检查文件路径
   - 验证数据形状一致性

3. **标准化问题**
   - 确保训练和测试使用相同的标准化参数

### 调试工具
```python
# 检查数据信息
info = loader.get_dataset_info()
print(f"训练样本: {info['train_samples']}")
print(f"数据形状: {info['train_data_shape']}")
print(f"数据源: {info['train_data_sources']}")
```

## 最佳实践

### 1. 数据准备
- 使用NPY格式获得最佳性能
- 预先标准化数据并保存
- 确保数据形状一致性

### 2. 内存优化
- 对于大文件，考虑分批加载
- 使用合适的批次大小
- 监控内存使用情况

### 3. 测试验证
- 始终验证数据加载结果
- 检查标准化效果
- 验证批次数据完整性

### 4. MOOSE集成
- 使用标准MOOSE输出格式
- 确保参数文件完整性
- 验证仿真结果一致性

---

*最后更新：2025年9月1日*