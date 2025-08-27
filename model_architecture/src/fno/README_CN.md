# 傅里叶神经算子 (FNO) 模型

该目录包含了低代码深度学习平台中傅里叶神经算子(FNO)模型的实现。

## 概述

傅里叶神经算子(FNO)是一种深度学习架构，专门用于学习函数空间之间的映射。它使用傅里叶变换来高效地捕捉数据中的全局依赖关系，使其在求解偏微分方程(PDE)和其他具有复杂空间模式的问题时特别有效。

## 实现的模型

1. **FNO1DEncoder**: 1D FNO谱编码器
2. **FNO2DEncoder**: 2D FNO谱编码器
3. **FNO**: 支持1D和2D情况的完整FNO模型

## 主要特性

- 支持1D和2D域
- 使用FFT进行谱卷积以实现高效的全局操作
- 坐标特征增强
- 可配置的层数、模式数和通道数
- 与低代码平台的模型接口兼容

## 超参数

FNO模型支持以下超参数：

- `in_channels`: 输入通道数
- `out_channels`: 输出通道数
- `decoder_layers`: 解码器层数
- `decoder_layer_size`: 解码器层中的神经元数量
- `dimension`: 模型维度(1或2)
- `latent_channels`: 谱卷积中的潜在特征大小
- `num_fno_layers`: 谱卷积层数
- `num_fno_modes`: 在谱卷积中保留的傅里叶模式数
- `padding`: 谱卷积的域填充
- `padding_type`: 谱卷积的填充类型
- `coord_features`: 使用坐标网格作为附加特征图

## 使用方法

在低代码平台中使用FNO模型：

```python
from model_manager import ModelManager

# 初始化模型管理器
model_manager = ModelManager({})

# 创建FNO模型
fno_model = model_manager.create_model('fno', 
                                      in_channels=1,
                                      out_channels=1,
                                      dimension=2,
                                      latent_channels=32,
                                      num_fno_layers=4,
                                      num_fno_modes=16)
```

## 参考文献

- Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).