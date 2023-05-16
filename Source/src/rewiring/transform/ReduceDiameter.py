from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx, to_networkx, is_undirected, contains_self_loops

import networkx as nx
from typing import Optional, Any
from math import floor
import numpy as np
import random
import torch

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

        self._x = None
        self._y = None

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

        G = self._to_networkx(data)

        # If the graph is disconnected, then the rewiring first connects
        # all connected components, so that the rewiring can be ran
        components = list(nx.connected_components(G))
        if len(components) != 1:
            for component in components:
                if len(component) >= 1:
                    random_node = random.sample(component, 1)[0]
                    # Add edges from each disconnected node to the random node
                    for node in G.nodes():
                        if node not in component:
                            G.add_edge(node, random_node)
                    break

        if self._max_diam is not None:
            while nx.diameter(G) > self._max_diam:
                G = self._add_edge_shortest_path(G)
        else:
            while nx.density(G) < self._min_density:
                G = self._add_edge_shortest_path(G)

        return self._from_networkx(G)

    def _from_networkx(nx_graph):
        # Infer graph type (directed or undirected)
        is_directed = nx_graph.is_directed()

        # Get node features, labels, and indices
        node_features = torch.tensor([nx_graph.nodes[n]['feat'] for n in nx_graph.nodes])
        node_labels = torch.tensor([nx_graph.nodes[n]['label'] for n in nx_graph.nodes])
        node_indices = torch.tensor(list(nx_graph.nodes))

        # Get edge indices, edge attributes, and edge directions
        edge_index = torch.tensor(list(nx_graph.edges)).t().contiguous()
        edge_attr = torch.tensor([nx_graph.edges[u, v]['weight'] for u, v in nx_graph.edges])

        # Determine graph type and handle multiple edges
        if nx_graph.is_multigraph():
            graph_class = nx.MultiDiGraph if is_directed else nx.MultiGraph
            graph = graph_class()
        else:
            graph_class = nx.DiGraph if is_directed else nx.Graph
            graph = graph_class()

        # Add nodes with features and labels
        for i, feat in enumerate(node_features):
            graph.add_node(node_indices[i], feat=feat.item(), label=node_labels[i].item())

        # Add edges with attributes
        for u, v in edge_index.t().tolist():
            graph.add_edge(u, v, weight=edge_attr[(edge_index[0] == u) & (edge_index[1] == v)].item())

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, y=node_labels, edge_index=edge_index, edge_attr=edge_attr, is_directed=is_directed)
        data.node_indices = node_indices

        return data

    def _to_networkx(data):
        # Infer graph type (directed or undirected)
        is_directed = data.is_directed

        # Create empty NetworkX graph
        if data.is_multigraph:
            graph_class = nx.MultiDiGraph if is_directed else nx.MultiGraph
            nx_graph = graph_class()
        else:
            graph_class = nx.DiGraph if is_directed else nx.Graph
            nx_graph = graph_class()



        # Add nodes with features
        for i, feat in enumerate(data.x):
            nx_graph.add_node(data.node_indices[i], feat=feat.item())

        # Add edges with attributes
        for u, v in data.edge_index.t().tolist():
            edge_attrs = data.edge_attr[(data.edge_index[0] == u) & (data.edge_index[1] == v)]
            for attr in edge_attrs:
                nx_graph.add_edge(u, v, weight=attr.item())

        return nx_graph

    def _add_edge_shortest_path(self, G):
        r"""Finds the longest shortest path in :obj:`G` and inserts an edge between the start and end node
            of the path.
        Args:
            G (required): Any type of graph object from NetworkX, e.g. :obj:`Graph`, :obj:`DiGraph`, etc."""
        shortest_paths = dict(nx.all_pairs_shortest_path(G=G))
        longest_ind = (0,0)

        for node1, paths in shortest_paths.items():
            for node2, path in paths.items():
                if len(path) > len(shortest_paths[longest_ind[0]][longest_ind[1]]):
                    longest_ind = (node1, node2) 

        G.add_edge(*longest_ind)
        return G

    def _is_multigraph(edge_index: torch.Tensor=None, data: Data=None) -> bool:
        if data is not None:
            edge_index = data.edge_index
        elif edge_index is not None:
            pass
        else:
            raise RuntimeError("Either edge_index or data must be set!")
        
        num_cols = edge_index.size(1)
        for col1_ind in range(num_cols):
            for col2_ind in range(col1_ind + 1, num_cols):
                col1 = edge_index[:, col1_ind]
                col2 = edge_index[:, col2_ind]
                if torch.all(torch.eq(col1, col2)).item():
                    return True
        return False   


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'