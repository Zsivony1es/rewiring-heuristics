# Graph Dataset Generator and Graph Rewiring Test Framework

### A simple framework written in PyTorch Geometric to create parameterized pseudo-random datasets, on which different rewiring heuristics can be tested and used to train GNN models

### About
This project was created for my (Peter Ivony) bachelor thesis at the University of Vienna, written between March 1st and June 30th. 
My supervising professor during the project was <a href="https://ufind.univie.ac.at/en/person.html?id=111520">Nils Morten Kriege</a>.

### Execution
The program is meant to be ran from the command line in a venv/Conda environment.
Required libraries:
+ <a href="https://numpy.org/">NumPy</a>
+ <a href="https://pytorch.org/">PyTorch</a>
+ <a href="https://pytorch-geometric.readthedocs.io/en/latest/">PyTorch Geometric</a>
+ <a href="https://networkx.org/">NetworkX</a>

To run it, just simply execute `Source/src/main.py` with the required arguments, which are:
| Argument tag  | Required | Possible Values | Description |
|---|---|---|---|
| `--model_type` | Yes | GCN, GIN, GAT | Set the model type |
| `--rewiring` | No | ARE, RD, DW | Choose rewiring mechanism, if not set, the dataset won't be changed |
| `--generate_plots` | No | N/A | Including this tag will generate plots to `Source/out` |

For example to execute the program with a GAT model and the ReduceDiameter rewiring technique including plot generation, you would execute:
```bash
Source/src/main.py --model_type GAT --rewiring RD --generate_plots
```