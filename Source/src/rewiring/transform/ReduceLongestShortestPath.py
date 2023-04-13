from torch_geometric.data.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class ReduceLongestShortestPath(BaseTransform):
    r"""Graph rewiring heuristic, that adds an edge between the two nodes, 
    which are the furthest away from each other."""

    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'