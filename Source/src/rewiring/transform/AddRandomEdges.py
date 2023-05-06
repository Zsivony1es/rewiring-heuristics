from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import numpy as np
import torch
from math import ceil

class AddRandomEdges(BaseTransform):
    """Adds edges randomly to the input graph"""
    # Number of edges should be a percentage

    def __init__(self, min_density: float = 0.15, num_edges: int = None, allow_self_loops: bool = False):
        r"""Initialize the transformation.
        Args:
            min_density (float, default=0.15): Ratio between 0 and 1. Sets how dense the graph should be filled with edges,
                compared to the complete graph, e.g. 0.25 means that after running the algorithm, the resulting graph will include
                at least 25% of all the edges in the complete graph
            num_edges (int, optional): Sets exactly how many edges should be added to the graph.
                Setting this value will ignore the input for min_density and use this as a configuration instead.
            allow_self_loops (bool, default=False): Set whether the transformation should include edges with the same node
                as beginning and end points."""
        
        assert min_density > 0 and min_density <= 1, "Minimum density must be between 0 and 1!"
        
        super().__init__()
        if (num_edges != None):
            self._num_edges_add = num_edges
            self._min_density = None
        else:
            # Initialization deferred to __call__() since we don't yet know the amount of edges
            self._num_edges_add = None
            self._min_density = min_density
        self._rand_threshold = 0.1
        self._allow_self_loops = allow_self_loops

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()
        
        complete_graph_edge_count = (data.num_nodes * (data.num_nodes - 1))/2
        free_edge_spot_count = complete_graph_edge_count - data.num_edges

        if self._num_edges_add == None and self._min_density != None:
            self._num_edges_add = ceil(self._min_density * complete_graph_edge_count) - data.num_edges

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

            data.validate()
            return data
                
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

            data.validate()
            return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ratio_edges={self._min_density}, num_edges={self._num_edges_add}, allow_self_loops={self._allow_self_loops})"