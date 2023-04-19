from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import random
import numpy as np
import torch

class AddRandomEdges(BaseTransform):
    """Adds edges randomly to the input graph"""
    # Number of edges should be a percentage

    def __init__(self, num_edges_add: int, allow_self_loops: bool = False):
        super().__init__()
        self._num_edges_add = num_edges_add
        self._rand_threshold = 0.1
        self._allow_self_loops = allow_self_loops

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()
        
        complete_graph_edge_count = (data.num_nodes * (data.num_nodes - 1))/2
        free_edge_spot_count = complete_graph_edge_count - data.num_edges

        assert data.num_nodes > 1
        assert free_edge_spot_count >= self._num_edges_add

        node_indices = data.edge_index.flatten().unique()

        # Use random node selection approach if the graph is not very dense
        # Otherwise runtime would be long AND indeterministic
        # so we find all nodes that aren't connected and choose from them
        
        if (free_edge_spot_count / complete_graph_edge_count) > self._rand_threshold:
            
            while True:
                node1, node2 = node_indices[np.random.choice(node_indices.shape[0], size=2, replace=self._allow_self_loops)]
                if node1 != node2:
                    print()
        else:
            # Get edge set of complete graph
            all_possible_edges = torch.cartesian_prod(node_indices, node_indices)

            # Get edge set of complement graph
            mask = torch.ones(len(all_possible_edges), dtype=torch.bool)
            for i in range(len(data.edge_index)):
                mask &= ~(all_possible_edges == data.edge_index[i]).all(dim=1)
            nonexistant_edge_set = all_possible_edges[mask]

            # Filter out self-looping edges
            if not self._allow_self_loops:
                nonexistant_edge_set = nonexistant_edge_set[nonexistant_edge_set[:, 0] != nonexistant_edge_set[:, 1]]

            # Select the edges we want to add randomly
            edges_to_add = nonexistant_edge_set[np.random.choice(nonexistant_edge_set.shape[0], size=5, replace=False)]
            data.edge_index = torch.cat((data.edge_index, edges_to_add))
            return data


    def _map_node_index(self, normalized_idx: int) -> int:
        r"""Allows other parts of the class to index the graph nodes on a continuous interval from 0 to num_nodes-1,
        since by default the indexes in """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_edges={self._num_edges})"