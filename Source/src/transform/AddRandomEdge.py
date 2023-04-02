from torch_geometric.data.data import Data
from transform import GraphTransformer

class AddRandomEdge(GraphTransformer):
    def __init__(self) -> None:
        super().__init__()
        return

    def optimize(self, graph) -> Data:
        pass