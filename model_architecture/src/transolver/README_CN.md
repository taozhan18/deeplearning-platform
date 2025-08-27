# Transolver 模型

## 概述

Transolver 是一种专门设计用于求解偏微分方程(PDE)的基于Transformer的求解器。它是从 https://github.com/thuml/Transolver 修改而来的。

Transolver 的主要特性包括：
- 物理信息注意力机制
- 结构化网格处理能力
- 灵活的架构，适用于各种PDE问题

## 架构

Transolver 由几个关键组件组成：

1. **物理注意力**：为物理问题设计的专门注意力机制
2. **Transformer块**：带有物理信息修改的标准Transformer编码器块
3. **MLP层**：用于特征处理的多层感知机
4. **嵌入层**：位置和时间步长嵌入

## 主要特性

- 专为结构化2D网格数据设计
- 物理信息注意力机制
- 支持带时间步长嵌入的时间相关PDE
- 可配置的层数、头数和隐藏维度
- 支持多种激活函数

## 参数

| 参数 | 描述 | 类型 | 默认值 |
|------|------|------|--------|
| `space_dim` | 输入数据的空间维度 | int | 1 |
| `n_layers` | Transformer层数 | int | 5 |
| `n_hidden` | Transformer的隐藏维度 | int | 256 |
| `dropout` | Dropout率 | float | 0.0 |
| `n_head` | 注意力头数 | int | 8 |
| `Time_Input` | 是否包含时间嵌入 | bool | False |
| `act` | 激活函数 | str | 'gelu' |
| `mlp_ratio` | MLP中的隐藏维度比例 | int | 1 |
| `fun_dim` | 函数的维度 | int | 1 |
| `out_dim` | 输出维度 | int | 1 |
| `slice_num` | 结构化注意力中的切片数 | int | 32 |
| `ref` | 参考维度 | int | 8 |
| `unified_pos` | 是否使用统一位置嵌入 | bool | False |
| `H` | 网格的高度 | int | 85 |
| `W` | 网格的宽度 | int | 85 |

## 使用方法

在配置中使用Transolver模型：

```yaml
model:
  name: "transolver"
  parameters:
    space_dim: 2
    n_layers: 6
    n_hidden: 256
    dropout: 0.1
    n_head: 8
    Time_Input: False
    act: "gelu"
    mlp_ratio: 1
    fun_dim: 1
    out_dim: 1
    slice_num: 32
    ref: 8
    unified_pos: False
    H: 85
    W: 85
```

## 参考文献

1. Transolver 仓库: https://github.com/thuml/Transolver
2. MIT License Copyright (c) 2024 THUML @ Tsinghua University