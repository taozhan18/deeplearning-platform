# 多层感知机（MLP）模型

该目录包含了低代码深度学习平台的多层感知机（MLP）模型实现。

## 概述

多层感知机（MLP）是一类前馈人工神经网络，由至少三层节点组成：输入层、隐藏层和输出层。除了输入节点外，每个节点都是使用非线性激活函数的神经元。

## 实现的模型

1. **MLP**: 灵活的多层感知机实现

## 主要特性

- 可配置的隐藏层数量
- 支持每层不同的激活函数
- 支持跳跃连接以训练更深的网络
- 可配置的dropout率
- 灵活的层大小设置

## 超参数

MLP模型支持以下超参数：

- `in_features`: 输入特征大小
- `layer_sizes`: 隐藏层大小。可以是所有层的单个整数，也可以是每层的列表
- `out_features`: 输出特征大小
- `num_layers`: 隐藏层数量
- `activation_fn`: 使用的激活函数（支持'relu'、'tanh'、'sigmoid'、'leaky_relu'、'elu'、'selu'、'gelu'、'silu'、'none'）
- `skip_connections`: 每隔2个隐藏层添加跳跃连接
- `dropout`: Dropout率。可以是所有层的单个浮点数，也可以是每层的列表

## 使用方法

在低代码平台中使用MLP模型：

```python
from model_manager import ModelManager

# 初始化模型管理器
model_manager = ModelManager({})

# 创建MLP模型
mlp_model = model_manager.create_model('mlp', 
                                     in_features=784,
                                     layer_sizes=256,
                                     out_features=10,
                                     num_layers=4,
                                     activation_fn='relu',
                                     dropout=0.2)
```

## 支持的激活函数

- `relu`: 线性整流单元
- `tanh`: 双曲正切函数
- `sigmoid`: Sigmoid函数
- `leaky_relu`: 泄漏线性整流单元
- `elu`: 指数线性单元
- `selu`: 扩展指数线性单元
- `gelu`: 高斯误差线性单元
- `silu`: Sigmoid线性单元（Swish）
- `none`: 无激活函数

## 跳跃连接

可以启用跳跃连接来帮助训练更深的网络，通过允许梯度更容易地流经网络。启用时，每隔2个隐藏层添加跳跃连接。