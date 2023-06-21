from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_networkx, to_networkx, is_undirected, contains_self_loops

import networkx as nx
from typing import Optional, Any, Union
from math import floor
import numpy as np
import random
import torch

class ReduceDiameter(BaseTransform):
    r"""Graph rewiring heuristic, that adds an edge between the two nodes, which have the longest shortest path 
    between each other in the graph, effectively reducing the diameter of the graph."""

    # input: maximum diameter or percentage of edges added

    def __init__(self, max_diam: Optional[int] = None, min_density: Optional[float] = None) -> None:
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
        self._avg_edge_attr = None
        self.edges_added = 0

        assert not ((max_diam is None) and (min_density is None)), "Either the maximum diameter or a ratio to the amount of current edges must be specified!"
        if max_diam is not None:
            assert max_diam > 0, "Maximum diameter must be positive!"
            self._max_diam = max_diam
            self._min_density = None
        else:
            assert min_density > 0 and min_density <= 1, "Minimum density must be bigger than 0 and at most 1!"
            self._max_diam = None
            self._min_density = min_density

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:

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

    def _from_networkx(self, G) -> Data:
        """Converts a NetworkX graph into a :obj:`torch_geometric.data.data.Data` object."""

        if self._x is not None:
            x = self._x
        else:
            # Get node features
            x = torch.tensor([G.nodes[n]['attr'] for n in G.nodes])
            pass
        
        if self._y is not None:
            y = self._y
        else:
            y = None

        # Get edge indices, edge attributes, and edge directions
        if not G.is_directed():
            edge_list = []
            for u, v in G.edges():
                edge_list.append([u,v])
                edge_list.append([v,u])
        else:
            edge_list = [[u, v] for u, v in G.edges]
        edge_index = torch.tensor(edge_list).t().contiguous()

        has_attr_field = 'attr' in ([data for u, v, data in G.edges(data=True)][0].keys())
        if has_attr_field:
            edge_attrs = []
            parsed_multiedges = []
            for u, v in G.edges():
                multi_edge_attributes = G.get_edge_data(u, v)
                if 0 in multi_edge_attributes:
                    # Graph has multiple edges here, each has its own attribute
                    if (u, v) not in parsed_multiedges:
                        for key in multi_edge_attributes:
                            edge_attrs.append(multi_edge_attributes[key]['attr'])
                        parsed_multiedges.append((u,v))
                else:
                    if 'attr' in multi_edge_attributes:
                        edge_attrs.append(multi_edge_attributes['attr'])
                    else:
                        # For some reason a LRGB dataset has two edges (out of thousands) with no attribute data???
                        # This would be a reason to throw an error, instead to get it to work I just set it to [0,0]
                        edge_attrs.append([0,0])
                        # raise RuntimeError(f"The edge ({u},{v}) doesn't have any attributes assigned!")
            edge_attr = torch.tensor(edge_attrs)
            data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, is_directed=G.is_directed())
        else:
            data = Data(x=x, y=y, edge_index=edge_index, is_directed=G.is_directed())


        return data

    def _to_networkx(self, data: Data) -> Union[nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph]:
        """Converts a :obj:`torch_geometric.data.data.Data` object into a NetworkX graph."""

        # Infer NetworkX graph type
        is_directed = data.is_directed()
        if ReduceDiameter._is_multigraph(data=data):
            graph_class = nx.MultiDiGraph if is_directed else nx.MultiGraph
            nx_graph = graph_class()
        else:
            graph_class = nx.DiGraph if is_directed else nx.Graph
            nx_graph = graph_class()

        # Save attributes, which won't be affected by rewiring
        # They can be returned without any conversion in _from_networkx()
        if data.x is not None:
            self._x = data.x
        if data.y is not None:
            self._y = data.y

        # Add nodes with features
        for i, feat in enumerate(data.x):
            nx_graph.add_node(i, attr=feat.tolist())

        # Add edges with attributes (if they exist)
        # Also I average out all edge attributes which will be used when inserting new edges
        avg_edge_attribute = []

        if data.edge_attr is not None:
            edge_index = 0
            for u, v in data.edge_index.t().tolist():
                attribute = data.edge_attr[edge_index].tolist()
                avg_edge_attribute.append(attribute)
                nx_graph.add_edge(u, v, attr=attribute)
                edge_index += 1
        else:
            for u, v in data.edge_index.t().tolist():
                nx_graph.add_edge(u, v)

        if data.edge_attr is not None:
            avg_edge_attribute = np.array(avg_edge_attribute)
            avgs = []
            for i in range(avg_edge_attribute.shape[1]):
                avgs.append(avg_edge_attribute[:,i].mean())
            self._avg_edge_attr = avgs

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

        if self._avg_edge_attr is not None:
            G.add_edge(*longest_ind, attr=self._avg_edge_attr)
        else:
            G.add_edge(*longest_ind)

        self.edges_added += 1

        return G

    def _is_multigraph(edge_index: torch.Tensor=None, data: Data=None) -> bool:
        if data is not None:
            edge_index = data.edge_index
        elif edge_index is not None:
            pass
        else:
            raise RuntimeError("Either edge_index or data must be set!")

        num_cols = edge_index.size(1)
        if num_cols < 2:
            return False

        edge_list = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
        unique_edges = set(edge_list)

        # If the lengths don't match then we filtered out multiple edges, which means it is a MultiGraph
        return len(unique_edges) != len(edge_list)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'