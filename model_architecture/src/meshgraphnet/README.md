# MeshGraphNet Model

## Overview

MeshGraphNet is a graph neural network architecture designed for mesh-based physics simulations. It was originally proposed in the paper "Learning mesh-based simulation with graph networks" by Pfaff et al. (2020).

The key features of MeshGraphNet include:
- Encoder-processor-decoder architecture
- Message passing on graph structures
- Support for node and edge features
- Flexible configuration of network components

## Architecture

MeshGraphNet consists of three main components:

1. **Encoders**: Transform input node and edge features into hidden representations
2. **Processor**: Performs iterative message passing updates on the graph
3. **Decoder**: Maps the final node representations to the output space

## Key Features

- Designed for physics simulations on mesh-based domains
- Supports various graph structures through DGL
- Configurable number of processor steps
- Flexible MLP configurations for encoders, processors, and decoders
- Support for different aggregation functions

## Parameters

| Parameter | Description | Type | Default |
|----------|-------------|------|---------|
| `input_dim_nodes` | Number of node features | int | 4 |
| `input_dim_edges` | Number of edge features | int | 3 |
| `output_dim` | Number of outputs | int | 2 |
| `processor_size` | Number of message passing blocks | int | 15 |
| `mlp_activation_fn` | Activation function to use | str | 'relu' |
| `num_layers_node_processor` | Number of MLP layers for processing nodes in each message passing block | int | 2 |
| `num_layers_edge_processor` | Number of MLP layers for processing edge features in each message passing block | int | 2 |
| `hidden_dim_processor` | Hidden layer size for the message passing blocks | int | 128 |
| `hidden_dim_node_encoder` | Hidden layer size for the node feature encoder | int | 128 |
| `num_layers_node_encoder` | Number of MLP layers for the node feature encoder | int | 2 |
| `hidden_dim_edge_encoder` | Hidden layer size for the edge feature encoder | int | 128 |
| `num_layers_edge_encoder` | Number of MLP layers for the edge feature encoder | int | 2 |
| `hidden_dim_node_decoder` | Hidden layer size for the node feature decoder | int | 128 |
| `num_layers_node_decoder` | Number of MLP layers for the node feature decoder | int | 2 |
| `aggregation` | Message aggregation type | str | 'sum' |

## Usage

To use the MeshGraphNet model in your configuration:

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

## References

1. Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks." arXiv preprint arXiv:2010.03409 (2020).
2. DGL documentation: https://www.dgl.ai/