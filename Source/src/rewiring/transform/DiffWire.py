from torch_geometric.data.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class DiffWire(BaseTransform):
    """Rewiring heuristic as described in the paper: DiffWire: Inductive Graph Rewiring via the Lovász Bound (https://openreview.net/pdf?id=IXvfIex0mX6f)"""

    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'