from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import random
import numpy as np
import torch
from math import floor

class AddRandomEdges(BaseTransform):
    """Adds edges randomly to the input graph"""
    # Number of edges should be a percentage

    def __init__(self, ratio_edges: float, num_edges: int = None, allow_self_loops: bool = False):
        r"""Initialize the transformation.
        Args:
            ratio_edges (float, default=0.05): Ratio between 0 and 1, of the current number of edges in the graph.
                If the result of multiplying is not a whole number, the program will round down to the nearest integer.
            num_edges (int, optional): Sets exactly how many edges should be added to the graph.
                Setting this value will ignore the input for ratio_edges and use this as a configuration instead.
            allow_self_loops (bool, default=False): Set whether the transformation should include edges with the same node
                as beginning and end points."""
        
        assert ratio_edges > 0
        assert ratio_edges < 1
        
        super().__init__()
        if (num_edges != None):
            self._num_edges_add = num_edges
            self._ratio_edges = None
        else:
            # Initialization deferred to __call__() since we don't yet know the amount of edges
            self._num_edges_add = None
            self._ratio_edges = ratio_edges
        self._rand_threshold = 0.1
        self._allow_self_loops = allow_self_loops

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()

        if self._num_edges_add == None and self._ratio_edges != None:
            self._num_edges_add = floor(self._ratio_edges * data.edge_index.shape[0])
        
        complete_graph_edge_count = (data.num_nodes * (data.num_nodes - 1))/2
        free_edge_spot_count = complete_graph_edge_count - data.num_edges

        assert data.num_nodes > 1
        assert free_edge_spot_count >= self._num_edges_add

        node_indices = data.edge_index.flatten().unique()

        # Use random node selection approach if the graph is not very dense
        # Otherwise runtime would be long AND indeterministic
        # so we find all nodes that aren't connected and choose from them
        
        if (free_edge_spot_count / complete_graph_edge_count) > self._rand_threshold:
            
            for _ in range(self._num_edges_add):
                while True:
                    # Get two node indices at random
                    edge = node_indices[np.random.choice(node_indices.shape[0], size=2, replace=self._allow_self_loops)]
                    # If the edge is not contained in the graph, we can break the loop and add it
                    if not torch.eq(edge, data.edge_index).all(1).any().item():
                        break
                # Add edge to graph
                data.edge_index = torch.cat((data.edge_index, edge.unsqueeze(0)), dim=0)
                
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