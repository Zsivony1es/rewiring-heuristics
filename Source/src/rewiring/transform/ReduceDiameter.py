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

    def __init__(self, max_diam: Optional[int] = None, min_density: Optional[float] = None, allow_self_loops: bool = False) -> None:
        r"""Initialize the transformation.
        Args:
            max_diam (int, optional): Continues adding edges into the graph, until the graph diameter is reduced to 
                at most this value.
            ratio_edges (float, optional): Ratio between 0 and 1. Sets how dense the graph should be filled with edges,
                compared to the complete graph, e.g. 0.25 means that after running the algorithm, the resulting graph will include
                at least 25% of all the edges in the complete graph
            allow_self_loops (bool, default=False): Set whether the transformation should include edges with the same node
                as beginning and end points."""
        
        super().__init__()
        assert not ((max_diam is None) and (min_density is None)), "Either the maximum diameter or a ratio to the amount of current edges must be specified!"
        if max_diam is not None:
            assert max_diam > 0, "Maximum diameter must be positive!"
            self._max_diam = max_diam
            self._min_density = None
        else:
            assert min_density > 0 and min_density <= 1, "Minimum density must be bigger than 0 and at most 1!"
            self._max_diam = None
            self._min_density = min_density
    

    def __call__(self, data: Data | HeteroData) -> Data | HeteroData:

        data.validate()

        G = to_networkx(data, to_undirected=is_undirected(data.edge_index))
        if self._max_diam is not None:
            while nx.diameter(G) > self._max_diam:
                shortest_paths = dict(nx.all_pairs_shortest_path(G=G))
                longest_ind = (0,0)

                for node1, paths in shortest_paths.items():
                    for node2, path in paths.items():
                        if len(path) > len(shortest_paths[longest_ind[0]][longest_ind[1]]):
                            longest_ind = (node1, node2) 

                G.add_edge(*longest_ind)
        else:
            while nx.density(G) < self._min_density:
                pass



    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'