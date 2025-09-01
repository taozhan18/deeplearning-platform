# 测试套件 - Test Suite

## 项目概述

本项目测试套件基于实际目录结构设计，用于验证深度学习平台的各个模块功能。测试采用PyTest框架，覆盖单元测试、集成测试和端到端验证。

## 实际目录结构

```
test/
├── README.md                          # 本说明文档
├── run_all_tests.py                  # 一键运行所有测试（已存在）
├── data_loader/                     # 数据加载测试
│   ├── test_data_loader.py          # 数据加载核心测试
│   ├── integration_test.py          # 数据集成测试
│   └── run_tests.py                 # 数据模块测试运行器
├── integration/                     # 端到端集成测试
│   ├── test_integration.py          # 主集成测试
│   ├── test_simple_integration.py   # 简化集成测试
│   ├── README.md                    # 集成测试说明
│   ├── fno/                        # FNO集成测试
│   ├── unet/                       # UNet集成测试
│   ├── transformer/                # Transformer集成测试
│   ├── transolver/                 # Transolver集成测试
│   ├── mlp/                        # MLP集成测试
│   └── meshgraphnet/               # MeshGraphNet集成测试
├── model/                          # 模型架构测试
│   ├── test_fno.py                 # FNO模型测试
│   ├── test_unet.py                # UNet测试
│   ├── test_transformer.py         # Transformer测试
│   ├── test_transolver.py          # Transolver测试
│   ├── test_mlp.py                 # MLP测试
│   ├── test_meshgraphnet.py        # MeshGraphNet测试
│   └── test_platform.py            # 平台通用测试
├── training_engine/                # 训练引擎测试
│   ├── test_training_engine.py     # 训练引擎核心测试
│   └── test_custom_preprocessing.py # 自定义预处理测试
└── moose/                          # MOOSE框架集成测试
    ├── test_moose_ml.py            # MOOSE机器学习测试
    ├── moose_diffusion_template.i  # MOOSE输入模板
    ├── moose_ml_config.json        # MOOSE配置
    ├── moose_ml_dataset/           # MOOSE测试数据集
    └── moose_simulations/          # MOOSE仿真数据
```

## 快速开始

### 安装测试依赖
```bash
pip install pytest pytest-cov
```

### 运行测试
```bash
# 一键运行所有测试
python test/run_all_tests.py

# 运行特定模块测试
python -m pytest test/data_loader/ -v
python -m pytest test/model/ -v
python -m pytest test/training_engine/ -v

# 运行集成测试
python -m pytest test/integration/ -v

# 运行MOOSE测试（需要MOOSE环境）
python -m pytest test/moose/ -v
```

## 测试模块详解

### 1. data_loader/ - 数据加载测试

**现有测试文件：**
- `test_data_loader.py` - 数据加载核心功能测试
- `integration_test.py` - 数据管道集成验证
- `run_tests.py` - 数据模块独立测试运行器

**测试覆盖：**
- CSV/JSON/NPY/NPZ格式数据加载
- 多源数据融合测试
- 数据标准化验证
- 批次处理和DataLoader集成

**运行示例：**
```bash
cd test/data_loader
python run_tests.py
```

### 2. model/ - 模型架构测试

**现有模型测试：**
- `test_fno.py` - 傅里叶神经算子（FNO）测试
- `test_unet.py` - UNet架构测试
- `test_transformer.py` - Transformer模型测试
- `test_transolver.py` - Transolver科学计算模型测试
- `test_mlp.py` - 多层感知机测试
- `test_meshgraphnet.py` - 图神经网络测试
- `test_platform.py` - 平台通用功能测试

**测试内容：**
```python
# 示例：FNO模型测试
def test_fno_forward_pass():
    model = create_model("fno", in_channels=3, out_channels=1, modes=12)
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    assert output.shape == (1, 1, 64, 64)
```

### 3. training_engine/ - 训练引擎测试

**现有测试：**
- `test_training_engine.py` - 训练循环核心功能
- `test_custom_preprocessing.py` - 自定义预处理验证

**测试覆盖：**
- 优化器配置（Adam, SGD, AdamW）
- 损失函数测试（MSE, CrossEntropy等）
- 学习率调度器验证
- 模型保存和加载
- 自定义预处理函数集成

### 4. integration/ - 集成测试

**实际集成测试：**
- `test_integration.py` - 主集成测试
- `test_simple_integration.py` - 简化版本测试
- 每个模型都有独立的配置和预处理文件

**集成测试结构：**
```
integration/[model_name]/
├── config_[model_name].yaml    # 模型特定配置
└── preprocess_[model_name].py  # 模型特定预处理
```

**运行示例：**
```bash
# 运行FNO集成测试
python test/integration/test_integration.py --config integration/fno/config_fno.yaml
```

### 5. moose/ - MOOSE框架测试

**现有MOOSE测试：**
- `test_moose_ml.py` - MOOSE与机器学习集成测试
- 包含完整的MOOSE仿真数据集
- 支持参数-结果映射验证

**MOOSE数据集：**
- `moose_ml_dataset/` - 包含训练好的模型和配置
- `moose_simulations/` - 10个MOOSE仿真案例
- `moose_ml_config.json` - MOOSE机器学习配置

## 测试数据

### 1. 测试数据位置
- 集成测试使用模拟数据
- 实际测试数据在运行时生成
- MOOSE测试使用真实仿真数据

### 2. 测试配置模板
每个模型在integration/下都有对应的配置文件：
```yaml
# integration/fno/config_fno.yaml
model:
  name: "fno"
  parameters:
    in_channels: 3
    out_channels: 1
    modes: 12
    width: 32
```

## 开发计划

### 短期计划（1-2个月）
1. **测试覆盖率提升**
   - 添加边界条件测试
   - 增加异常处理测试

2. **性能测试增强**
   - 添加内存使用监控
   - 增加训练速度基准测试

3. **测试文档完善**
   - 为每个模型添加详细测试说明
   - 创建测试最佳实践指南

### 中长期计划（3-6个月）
1. **测试自动化**
   - 集成GitHub Actions CI/CD
   - 自动测试报告生成

2. **测试数据标准化**
   - 统一测试数据格式
   - 创建测试数据生成工具

3. **高级测试功能**
   - 添加回归测试套件
   - 实现性能回归监控

## 故障排除

### 常见问题
1. **测试数据缺失** - 运行集成测试前确保配置正确
2. **路径问题** - 设置PYTHONPATH环境变量
3. **依赖缺失** - 安装pytest和相关测试库

### 调试工具
```bash
# 调试特定测试
python -m pytest test/model/test_fno.py --pdb

# 查看详细输出
python -m pytest test/integration/ -v --tb=long
```

## 测试维护

### 定期维护任务
- 每月检查测试通过情况
- 更新测试配置以匹配代码变更
- 清理过时的测试文件
- 监控测试执行时间

### 测试扩展指南
1. 为新模型添加测试：在model/下创建test_[model].py
2. 为新功能添加集成测试：在integration/下添加配置和预处理文件
3. 更新run_all_tests.py以包含新测试模块

---

*最后更新：2025年9月1日*