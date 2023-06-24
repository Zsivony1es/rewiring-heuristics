from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import is_undirected

import numpy as np
import networkx as nx
import torch
import random
from typing import Union
from math import ceil
import logging

log = logging.getLogger(__name__)
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

        self._x = None
        self._y = None
        self._avg_edge_attr = None
        self.edges_added = 0

    def __call__(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        data.validate()
        log.debug("Adding edges...")
        G = self._to_networkx(data=data)
        
        nodes = list(G.nodes)
        existing_edges = set(G.edges)
        num_nodes = len(nodes)

        if data.is_undirected:
            max_possible_edges = int((data.num_nodes * (num_nodes - 1)) / 2)
        else:
            max_possible_edges = int(data.num_nodes * (data.num_nodes - 1))
            
        free_edge_spot_count = max_possible_edges - data.num_edges

        if self._num_edges_add == None and self._min_density != None:
            num_edges_add = int(ceil(self._min_density * max_possible_edges) - data.num_edges)

        if num_edges_add < 1:
            return data

        assert data.num_nodes > 1, "Graph only contains one node!"
        assert free_edge_spot_count >= num_edges_add, \
            f"""There are not enough free edge spots in the graph! [{data.num_edges}/{max_possible_edges}] \
You want to add {num_edges_add} edges, but only {free_edge_spot_count} spots remain! {data.num_nodes}"""
        
        new_num_edges = data.num_edges + num_edges_add
        
        if new_num_edges / max_possible_edges > 0.3:
            # If the graph is dense enough, then randomly sampling edges won't be as successful
            # So I compute the complement graph and add the new edges from that.
            G_c = nx.complement(G)
            new_edges = random.sample(list(G_c.edges), num_edges_add)
            for u, v in new_edges:
                G.add_edge(u, v)
        else:
            while len(existing_edges) < new_num_edges:
                node1, node2 = random.sample(nodes, 2)
                if (node1, node2) not in existing_edges and (node2, node1) not in existing_edges:
                    G.add_edge(node1, node2)
                    existing_edges.add((node1, node2))
        
        data = self._from_networkx(G=G)
        data.validate()
        
        return data
    
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
        if AddRandomEdges._is_multigraph(data=data):
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
        return f"{self.__class__.__name__}(ratio_edges={self._min_density}, num_edges={self._num_edges_add}, allow_self_loops={self._allow_self_loops})"