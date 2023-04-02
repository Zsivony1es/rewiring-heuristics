from torch_geometric.data.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class AddRandomEdge(BaseTransform):
    """Adds edges randomly to the input graph"""

    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'