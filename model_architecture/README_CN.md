# 模型架构模块

模型架构模块提供了基于PyTorch的预定义深度学习模型集合。

## 功能特点

- 预定义的流行模型架构（ResNet、VGG、LSTM等）
- 每个模型都有清晰的超参数描述
- 定义新模型架构的模板
- 简单的模型选择和配置
- 傅里叶神经算子(FNO)模型，适用于PDE相关任务

## 使用方法

用户可以从预定义模型中选择，或使用我们的模板定义自己的架构。每个模型都附有详细的超参数描述，帮助用户理解其功能。

## 模型模板

我们提供了一个标准模板用于实现新的模型架构。所有模型都应遵循此模板，以确保一致性和易用性。

## FNO模型

该模块现在包含了傅里叶神经算子(FNO)模型，对于求解偏微分方程和其他具有复杂空间模式的问题特别有效。详见[FNO文档](src/fno/README_CN.md)了解更多详情。

## 可用模型

- `linear`, `conv2d`, `lstm`, `gru`, `rnn` - 基础PyTorch层
- `relu`, `sigmoid`, `tanh`, `softmax` - 激活函数
- `modeltemplate` - 基础模型模板
- `fno` - 傅里叶神经算子模型(1D和2D)

通过编程方式获取所有可用模型列表：

```python
from model_manager import ModelManager
model_manager = ModelManager({})
print(model_manager.get_available_models())
```