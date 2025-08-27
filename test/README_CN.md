# 测试套件

该目录包含低代码深度学习平台的测试套件。

## 目录结构

```
test/
├── data_loader/           # 数据加载模块的测试
├── model/                 # 模型实现的测试
├── training_engine/       # 训练引擎模块的测试
├── run_tests.py           # 主测试运行脚本
└── README.md              # 本文件
```

## 测试组织

### 数据加载测试
数据加载和预处理功能的测试：
- `test_data_loader.py` - 数据加载器基本功能测试

### 模型测试
模型实现的测试：
- `test_platform.py` - 平台集成测试
- `test_fno.py` - FNO模型实现的专门测试

### 训练引擎测试
训练引擎功能的测试：
- `test_training_engine.py` - 训练引擎组件的测试

## 运行测试

要运行所有测试，请执行主测试运行器：

```bash
python test/run_tests.py
```

这将按顺序运行所有测试脚本并提供结果摘要。

## 单独执行测试

您也可以直接运行单个测试脚本：

```bash
# 运行平台集成测试
python test/model/test_platform.py

# 运行FNO模型测试
python test/model/test_fno.py

# 运行数据加载器测试
python test/data_loader/test_data_loader.py

# 运行训练引擎测试
python test/training_engine/test_training_engine.py
```

## 测试环境

所有测试都应在包含所有必要依赖项的`physics` conda环境中运行。