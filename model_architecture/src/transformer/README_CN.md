# Transformer 模型

## 概述

Transformer 模型是一种强大的序列到序列任务架构，最初在 Vaswani 等人的论文《Attention Is All You Need》中提出。该实现提供了一个灵活的 Transformer 架构，适用于各种序列建模任务。

## 架构

Transformer 由编码器和解码器组成：

1. **编码器**：由一组相同的层堆叠而成，每层包含：
   - 多头自注意力机制
   - 位置前馈网络
   - 残差连接和层归一化

2. **解码器**：也由一组相同的层堆叠而成，每层包含：
   - 掩码多头自注意力
   - 对编码器输出的多头注意力
   - 位置前馈网络
   - 残差连接和层归一化

3. **位置编码**：由于 Transformer 不包含循环或卷积，因此使用正弦函数注入位置信息。

## 主要特性

- 可配置的层数、注意力头数和隐藏维度
- 支持可变序列长度
- 用于正则化的 Dropout
- 用于训练稳定性的层归一化

## 参数

| 参数 | 描述 | 类型 | 默认值 |
|------|------|------|--------|
| `input_dim` | 输入特征维度 | int | 必需 |
| `output_dim` | 输出特征维度 | int | 必需 |
| `d_model` | 模型嵌入维度 | int | 512 |
| `n_layers` | 编码器/解码器层数 | int | 6 |
| `n_heads` | 注意力头数 | int | 8 |
| `pf_dim` | 位置前馈网络隐藏维度 | int | 2048 |
| `dropout` | Dropout 率 | float | 0.1 |
| `max_len` | 最大序列长度 | int | 100 |

## 使用方法

在配置中使用 Transformer 模型：

```yaml
model:
  name: "transformer"
  parameters:
    input_dim: 10
    output_dim: 5
    d_model: 256
    n_layers: 4
    n_heads: 8
    pf_dim: 1024
    dropout: 0.1
    max_len: 50
```

## 参考文献

1. Vaswani, A., et al. "Attention is all you need." Advances in Neural Information Processing Systems 30 (2017).