# 低代码深度学习平台模型指南

本文档提供了平台中所有可用的完整神经网络模型及其参数的信息。

## 可用的完整模型

平台目前支持以下完整神经网络架构：

1. **自定义模型**：
   - `modeltemplate`：基础前馈神经网络模板
   - `fno`：傅里叶神经算子，适用于PDE相关任务
   - `mlp`：多层感知机，具有可配置的层数和激活函数
   - `unet`：用于图像分割任务的U-Net

注意：平台还包含基础PyTorch层（如Linear、Conv2d、LSTM等）和激活函数（如ReLU、Sigmoid等），这些可以用来构建自定义架构，但它们本身不是完整的模型。
```

/home/zt/workspace/deeplearning-platform/MODEL_GUIDE_CN.md
```markdown
<<<<<<< SEARCH
## 完整模型参数

### 1. ModelTemplate

一个具有可配置层和dropout的基础前馈神经网络。

**参数**：
- `input_size` (int, 默认: 784)：输入特征的大小
- `hidden_size` (int, 默认: 256)：隐藏层中的单元数
- `num_classes` (int, 默认: 10)：输出类别的数量
- `dropout_rate` (float, 默认: 0.2)：用于正则化的dropout率

**配置示例**：
```yaml
model:
  name: "ModelTemplate"
  parameters:
    input_size: 784
    hidden_size: 256
    num_classes: 10
    dropout_rate: 0.2
```

### 2. FNO (傅里叶神经算子)

一种为学习函数空间之间的映射而设计的模型，对于PDE相关任务特别有效。

**参数**：
- `in_channels` (int, 默认: 1)：输入通道数
- `out_channels` (int, 默认: 1)：输出通道数
- `decoder_layers` (int, 默认: 1)：解码器层数
- `decoder_layer_size` (int, 默认: 32)：解码器层中的神经元数
- `dimension` (int, 默认: 2)：模型维度（1表示1D，2表示2D）
- `latent_channels` (int, 默认: 32)：谱卷积中的潜在特征大小
- `num_fno_layers` (int, 默认: 4)：谱卷积层数
- `num_fno_modes` (Union[int, List[int]], 默认: 16)：在谱卷积中保留的傅里叶模式数
- `padding` (int, 默认: 8)：谱卷积的域填充
- `padding_type` (str, 默认: "constant")：谱卷积的填充类型
- `coord_features` (bool, 默认: True)：使用坐标网格作为附加特征图

**FNO1D配置示例**：
```yaml
model:
  name: "fno"
  parameters:
    in_channels: 1
    out_channels: 1
    dimension: 1
    latent_channels: 16
    num_fno_layers: 3
    num_fno_modes: 16
```

**FNO2D配置示例**：
```yaml
model:
  name: "fno"
  parameters:
    in_channels: 3
    out_channels: 2
    dimension: 2
    latent_channels: 32
    num_fno_layers: 4
    num_fno_modes: [16, 16]
```

### 3. MLP (多层感知机)

一个灵活的多层感知机实现，具有可配置的层数、激活函数和跳跃连接。

**参数**：
- `in_features` (int, 默认: 512)：输入特征大小
- `layer_sizes` (Union[int, List[int]], 默认: 512)：隐藏层大小。可以是所有层的单个整数，也可以是每层的列表
- `out_features` (int, 默认: 512)：输出特征大小
- `num_layers` (int, 默认: 6)：隐藏层数量
- `activation_fn` (Union[str, List[str]], 默认: 'relu')：使用的激活函数
- `skip_connections` (bool, 默认: False)：每隔2个隐藏层添加跳跃连接
- `dropout` (Union[float, List[float]], 默认: 0.0)：Dropout率。可以是所有层的单个浮点数，也可以是每层的列表

**支持的激活函数**：
- `relu`: 线性整流单元
- `tanh`: 双曲正切函数
- `sigmoid`: Sigmoid函数
- `leaky_relu`: 泄漏线性整流单元
- `elu`: 指数线性单元
- `selu`: 扩展指数线性单元
- `gelu`: 高斯误差线性单元
- `silu`: Sigmoid线性单元（Swish）
- `none`: 无激活函数

**配置示例**：
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: 256
    out_features: 10
    num_layers: 4
    activation_fn: "relu"
    dropout: 0.2
```

**具有可变层的配置示例**：
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: [512, 256, 128, 64]
    out_features: 10
    num_layers: 4
    activation_fn: ["relu", "tanh", "relu", "sigmoid"]
    dropout: [0.1, 0.2, 0.2, 0.3]
    skip_connections: True
```

### 4. UNet

用于图像分割任务的标准UNet实现，具有编码器-解码器架构和跳跃连接。

**参数**：
- `in_channels` (int, 默认: 3)：输入图像的通道数
- `out_channels` (int, 默认: 1)：输出分割图的通道数
- `features` (List[int], 默认: [64, 128, 256, 512, 1024])：各层级的特征通道数
- `activation` (str, 默认: 'relu')：使用的激活函数类型
- `normalization` (str, 默认: 'batchnorm')：使用的归一化类型

**支持的激活函数**：
- `relu`: 线性整流单元
- `leaky_relu`: 泄漏线性整流单元
- `elu`: 指数线性单元

**支持的归一化类型**：
- `batchnorm`: 批量归一化
- `groupnorm`: 组归一化

**配置示例**：
```yaml
model:
  name: "unet"
  parameters:
    in_channels: 3
    out_channels: 1
    features: [64, 128, 256, 512, 1024]
    activation: "relu"
    normalization: "batchnorm"
```

## 如何使用模型

要使用这些模型，请在配置文件中指定模型名称和参数：

```yaml
model:
  name: "model_name"  # 例如, "fno", "ModelTemplate", "mlp"
  parameters:
    # 模型特定参数
```

然后运行训练：
```bash
python main/train.py --config your_config.yaml
```
## 如何使用模型

要使用这些模型，请在配置文件中指定模型名称和参数：

```yaml
model:
  name: "model_name"  # 例如, "fno", "ModelTemplate"
  parameters:
    # 模型特定参数
```

然后运行训练：
```bash
python main/train.py --config your_config.yaml
```

### 3. MLP (多层感知机)

一个灵活的多层感知机实现，具有可配置的层数、激活函数和跳跃连接。

**参数**：
- `in_features` (int, 默认: 512)：输入特征大小
- `layer_sizes` (Union[int, List[int]], 默认: 512)：隐藏层大小。可以是所有层的单个整数，也可以是每层的列表
- `out_features` (int, 默认: 512)：输出特征大小
- `num_layers` (int, 默认: 6)：隐藏层数量
- `activation_fn` (Union[str, List[str]], 默认: 'relu')：使用的激活函数
- `skip_connections` (bool, 默认: False)：每隔2个隐藏层添加跳跃连接
- `dropout` (Union[float, List[float]], 默认: 0.0)：Dropout率。可以是所有层的单个浮点数，也可以是每层的列表

**支持的激活函数**：
- `relu`: 线性整流单元
- `tanh`: 双曲正切函数
- `sigmoid`: Sigmoid函数
- `leaky_relu`: 泄漏线性整流单元
- `elu`: 指数线性单元
- `selu`: 扩展指数线性单元
- `gelu`: 高斯误差线性单元
- `silu`: Sigmoid线性单元（Swish）
- `none`: 无激活函数

**配置示例**：
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: 256
    out_features: 10
    num_layers: 4
    activation_fn: "relu"
    dropout: 0.2
```

**具有可变层的配置示例**：
```yaml
model:
  name: "mlp"
  parameters:
    in_features: 784
    layer_sizes: [512, 256, 128, 64]
    out_features: 10
    num_layers: 4
    activation_fn: ["relu", "tanh", "relu", "sigmoid"]
    dropout: [0.1, 0.2, 0.2, 0.3]
    skip_connections: True
```

### 4. 基础PyTorch层

这些是标准的PyTorch层。有关详细参数信息，请参阅[PyTorch文档](https://pytorch.org/docs/stable/index.html)。

**线性层的常见参数**：
- `in_features`：每个输入样本的大小
- `out_features`：每个输出样本的大小
- `bias`：如果设置为False，该层将不学习加性偏置

**Conv2d层的常见参数**：
- `in_channels`：输入图像中的通道数
- `out_channels`：卷积产生的通道数
- `kernel_size`：卷积核的大小

## 如何使用模型

要使用这些模型，请在配置文件中指定模型名称和参数：

```yaml
model:
  name: "model_name"  # 例如, "fno", "ModelTemplate", "linear"
  parameters:
    # 模型特定参数
```

然后运行训练：
```bash
python main/train.py --config your_config.yaml
```

## 添加新模型

要向平台添加新模型：
1. 实现您的模型类
2. 在ModelManager中注册它
3. 按照模板模式添加超参数文档
4. 使用平台测试模型

请参阅`model_architecture/src/`中的现有模型作为示例。