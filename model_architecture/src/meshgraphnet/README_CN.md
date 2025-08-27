# MeshGraphNet 模型

## 概述

MeshGraphNet 是一种专为基于网格的物理仿真设计的图神经网络架构。它最初在 Pfaff 等人（2020年）的论文"Learning mesh-based simulation with graph networks"中提出。

MeshGraphNet 的主要特性包括：
- 编码器-处理器-解码器架构
- 图结构上的消息传递
- 支持节点和边特征
- 网络组件的灵活配置

## 架构

MeshGraphNet 由三个主要组件组成：

1. **编码器**：将输入的节点和边特征转换为隐藏表示
2. **处理器**：在图上执行迭代的消息传递更新
3. **解码器**：将最终的节点表示映射到输出空间

## 主要特性

- 专为网格域上的物理仿真设计
- 通过DGL支持各种图结构
- 可配置的处理器步骤数量
- 编码器、处理器和解码器的灵活MLP配置
- 支持不同的聚合函数

## 参数

| 参数 | 描述 | 类型 | 默认值 |
|------|------|------|--------|
| `input_dim_nodes` | 节点特征数量 | int | 4 |
| `input_dim_edges` | 边特征数量 | int | 3 |
| `output_dim` | 输出数量 | int | 2 |
| `processor_size` | 消息传递块数量 | int | 15 |
| `mlp_activation_fn` | 使用的激活函数 | str | 'relu' |
| `num_layers_node_processor` | 每个消息传递块中处理节点的MLP层数 | int | 2 |
| `num_layers_edge_processor` | 每个消息传递块中处理边特征的MLP层数 | int | 2 |
| `hidden_dim_processor` | 消息传递块的隐藏层大小 | int | 128 |
| `hidden_dim_node_encoder` | 节点特征编码器的隐藏层大小 | int | 128 |
| `num_layers_node_encoder` | 节点特征编码器的MLP层数 | int | 2 |
| `hidden_dim_edge_encoder` | 边特征编码器的隐藏层大小 | int | 128 |
| `num_layers_edge_encoder` | 边特征编码器的MLP层数 | int | 2 |
| `hidden_dim_node_decoder` | 节点特征解码器的隐藏层大小 | int | 128 |
| `num_layers_node_decoder` | 节点特征解码器的MLP层数 | int | 2 |
| `aggregation` | 消息聚合类型 | str | 'sum' |

## 使用方法

在配置中使用MeshGraphNet模型：

```yaml
model:
  name: "meshgraphnet"
  parameters:
    input_dim_nodes: 4
    input_dim_edges: 3
    output_dim: 2
    processor_size: 15
    mlp_activation_fn: "relu"
    num_layers_node_processor: 2
    num_layers_edge_processor: 2
    hidden_dim_processor: 128
    hidden_dim_node_encoder: 128
    num_layers_node_encoder: 2
    hidden_dim_edge_encoder: 128
    num_layers_edge_encoder: 2
    hidden_dim_node_decoder: 128
    num_layers_node_decoder: 2
    aggregation: "sum"
```

## 参考文献

1. Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks." arXiv preprint arXiv:2010.03409 (2020).
2. DGL 文档: https://www.dgl.ai/