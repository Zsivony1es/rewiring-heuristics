from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, is_undirected, contains_self_loops

import networkx as nx
from typing import Optional
from math import floor

class ReduceDiameter(BaseTransform):
    r"""Graph rewiring heuristic, that adds an edge between the two nodes, which have the longest shortest path 
    between each other in the graph, effectively reducing the diameter of the graph."""

    # input: maximum diameter or percentage of edges added

    def __init__(self, max_diam: Optional[int] = None, ratio_edges: Optional[float] = None, allow_self_loops: bool = False) -> None:
        r"""Initialize the transformation.
        Args:
            max_diam (int, optional): Continues adding edges into the graph, until the graph diameter is reduced to 
                at most this value.
            ratio_edges (float, optional): Ratio between 0 and 1, of the current number of edges in the graph.
                If the result of multiplying is not a whole number, the program will round down to the nearest integer.
            allow_self_loops (bool, default=False): Set whether the transformation should include edges with the same node
                as beginning and end points."""
        
        super().__init__()
        assert not ((max_diam is None) and (ratio_edges is None)), "Either the maximum diameter or a ratio to the amount of current edges must be specified!"
        if max_diam is not None:
            assert max_diam > 0, "Maximum diameter must be positive!"
            self._max_diam = max_diam
            self._ratio_edges = None
        else:
            assert ratio_edges > 0, "Ratio must be larger than 0!"
            self._max_diam = None
            self._ratio_edges = ratio_edges
    

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()

        G = to_networkx(data, to_undirected=is_undirected(data.edge_index))
        if self._max_diam is not None:
            while nx.diameter(G) > self._max_diam:
                pass
        else:
            floor(self._ratio_edges * data.edge_index.shape[0])



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'