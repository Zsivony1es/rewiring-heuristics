from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import random

class AddRandomEdges(BaseTransform):
    """Adds edges randomly to the input graph"""

    def __init__(self, num_edges_add: int):
        super().__init__()
        self._num_edges_add = num_edges_add
        self._rand_threshold = 0.1

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()
        
        complete_graph_edge_count = (data.num_nodes * (data.num_nodes - 1))/2
        free_edge_spot_count = complete_graph_edge_count - data.num_edges

        assert data.num_nodes > 1
        assert free_edge_spot_count >= self._num_edges_add

        # Use random node selection approach if the graph is not very dense
        # Otherwise runtime would be long AND indeterministic
        # so we find all nodes that aren't connected and choose from them

        if (free_edge_spot_count / complete_graph_edge_count) > self._rand_threshold:
            while True:
                node1, node2 = random.randrange(data.num_nodes), random.randrange(data.num_nodes)
                if node1 != node2:
                   pass     
        else:
            pass

        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_edges={self._num_edges})"