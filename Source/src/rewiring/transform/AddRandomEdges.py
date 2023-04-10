from torch_geometric.data.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class AddRandomEdges(BaseTransform):
    """Adds edges randomly to the input graph"""

    def __init__(self, num_edges: int):
        super().__init__()
        self._num_edges = num_edges

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:
        # assert that _num_edges is smaller than the number of possible edges that can be added to the graph
        pass