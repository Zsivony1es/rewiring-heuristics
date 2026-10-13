# Graph Dataset Generator and Graph Rewiring Test Framework

A framework built on PyTorch Geometric for generating parameterised pseudo-random graph datasets and for benchmarking graph rewiring heuristics as a preprocessing step when training GNN models.

## About

A university project at the University of Vienna, developed between March and August 2023 and supervised by [Nils Morten Kriege](https://ufind.univie.ac.at/en/person.html?id=111520).

Graph rewiring alters the edge set of a graph before message passing, with the aim of reducing over-squashing and over-smoothing in graph neural networks. This repository provides three things:

- a generator for synthetic sparse graph datasets with a configurable graph count, size and density,
- rewiring heuristics implemented as PyTorch Geometric transforms, usable as a `pre_transform`,
- a training and evaluation pipeline that trains GCN, GAT or GIN models on node- and graph-classification datasets and reports accuracy and loss curves.

## Rewiring heuristics

| Flag | Transform | Description |
|---|---|---|
| `ARE` | `AddRandomEdges` | Adds random edges until the graph reaches a minimum edge density. |
| `RD` | `ReduceDiameter` | Inserts edges until the graph diameter drops below a maximum. |
| `GDC` | `GDC` (PyTorch Geometric) | Graph Diffusion Convolution, a diffusion-based rewiring. |

Omitting `--rewiring` trains on the unmodified dataset, which serves as the baseline.

## Requirements

Run the project from a virtualenv or Conda environment with:

- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [NetworkX](https://networkx.org/)

## Datasets

`cora`, `citeseer`, `proteins`, `mutag`, `enzymes` and `lrgb-pascal` are downloaded automatically by PyTorch Geometric into `Source/res/datasets` on first use.

The `synthetic` dataset is not stored in this repository and has to be generated first:

```bash
python Source/src/GenerateSparseDatasetFilesApp.py
```

This writes `Source/res/datasets/synthetic/raw_dset_list.pt`; the processed files are created on the first load.

## Running

Both entry points resolve dataset paths relative to the working directory, so run them from the repository root.

```bash
python Source/src/MainApp.py --dataset_list 'cora,enzymes' --model_type GAT --rewiring RD --generate_plots
```

| Argument | Required | Values | Description |
|---|---|---|---|
| `--dataset_list` | Yes | `*` or a comma-separated list of `cora`, `citeseer`, `proteins`, `mutag`, `enzymes`, `lrgb-pascal`, `synthetic` | Datasets to train on. `*` uses every available dataset. Quote the value. |
| `--model_type` | Yes | `GCN`, `GAT`, `GIN` | GNN architecture to train. |
| `--rewiring` | No | `ARE`, `RD`, `GDC` | Rewiring heuristic. If unset, the dataset is left unchanged. |
| `--epochs` | No | integer, default `300` | Number of epochs per training cycle. |
| `--generate_plots` | No | flag | Writes loss and accuracy plots to `Source/out`. |

## Repository layout

| Path | Contents |
|---|---|
| `Source/src/MainApp.py` | CLI entry point for training and evaluation. |
| `Source/src/GenerateSparseDatasetFilesApp.py` | CLI entry point for generating the synthetic dataset. |
| `Source/src/graphgen/` | Synthetic sparse graph generation and the `SparseGraphDataset` wrapper. |
| `Source/src/rewiring/transform/` | The `AddRandomEdges` and `ReduceDiameter` rewiring transforms. |
| `Source/src/rewiring/pipeline/` | Dataset loading plus the node-, single-graph-node- and graph-classification pipelines. |
| `Source/src/rewiring/model/` | GCN and GAT model definitions. |
| `Source/src/notebooks/` | Experiment notebooks for the individual heuristics. |
