import torch
import random as rn
import numpy as np
from torch_geometric.data import Data
import networkx as nx


class GraphBuilder:
    """Used to generate a graph instance of :obj:`torch_geometric.data.Data`"""

    def __init__(self, conf: dict, rand_seed: int) -> None:
        assert conf is not None
        rn.seed(rand_seed)
        self._conf = conf

    def _get_base_graph(self, directed: bool, multiple_edges: bool) -> None:
        """Assign to :obj:`self._G` one of the four main graph types"""
        types = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
        ret_ind = int(directed) + (2 * int(multiple_edges))
        self._G = types[ret_ind]

    def _generate_graph_from_config(self, conf: dict) -> Data:
        """Create a graph based on the configuration received"""
        self._get_base_graph(directed=conf['directed'], multiple_edges=conf['multiple_edges'])
        self._generate_nodes(node_count=conf['node_count'], 
                             node_feature_dim=conf['node_feature_dim'])
        self._generate_edges(node_count=conf['node_count'], 
                             edge_count=conf['edge_count'], 
                             edge_feature_dim=conf['edge_feature_dim'], 
                             loop_edges=conf['loop_edges'], 
                             multiple_edges=conf['multiple_edges'])
        if conf['train_goal'] == 'node':
            self._generate_node_labels(node_count=conf['node_count'])
        elif conf['train_goal'] == 'graph':
            self._generate_graph_label()
        else:
            raise RuntimeError(f'Invalid configuration value for "train_goal" - {conf["train_goal"]}\nValid options: node, graph')

    def _generate_nodes(self, node_count: int, node_feature_dim: int) -> None:
        """Generate :obj:`node_count` number of nodes, each one with :obj:`node_feature_dim` attributes, which are integers between 0 and 10"""
        for node_ind in range(node_count):
            self._G.add_node(node_ind, attr=tuple(int(round(rn.random() * 10)) for _ in range(node_feature_dim)))

    def _generate_edges(self, node_count: int, edge_count: int, edge_feature_dim: int, loop_edges: bool, multiple_edges: bool) -> None:
        """Generate :obj:`edge_count` number of edges, each one with :obj:`edge_feature_dim` attributes, which are integers between 0 and 10"""
        added_edges = 0
        while added_edges != edge_count:
            generated_edge = tuple(np.random.choice(node_count, 2, replace=loop_edges))
            if not multiple_edges and self._G.has_edge(*generated_edge):
                continue
            self._G.add_edge(
                *generated_edge, 
                attr=tuple(int(round(rn.random() * 10)) for _ in range(edge_feature_dim))
            )
            added_edges += 1

    def _generate_node_labels(self, node_count: int) -> None:
        self._y = [int(round(rn.random())) for _ in range(node_count)]

    def _generate_graph_label(self) -> None:
        self._y = [int(round(rn.random()))]

    def build(self) -> Data:
        """Returns the graph as an instance of :obj:`torch_geometric.data.Data`."""
        
        self._generate_graph_from_config(conf=self._conf)

        # couldn't get torch_geometric.utils.from_networkx() working
        edges = [(edge[0], edge[1]) for edge in self._G.edges]
        edge_index = torch.tensor(edges).t().contiguous()
        num_nodes = self._G.number_of_nodes()
        x = torch.tensor([self._G.nodes[node]['attr'] for node in range(num_nodes)])
        edge_attr = torch.tensor([self._G.edges[edge]['attr'] for edge in self._G.edges])
        ret_val = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=self._y)

        assert ret_val.x.shape[0] == num_nodes
        assert ret_val.x.shape[1] == len(self._G.nodes[0]['attr'])
        assert ret_val.edge_index.shape[0] == 2, f'edge_index attribute should have 2 columns, but has {ret_val.edge_index.shape[0]}\nedge_index: {ret_val.edge_index}'
        assert ret_val.edge_index.shape[1] == len(edges)
        assert ret_val.edge_attr.shape[0] == len(edges)
        assert ret_val.edge_attr.shape[1] == len(edge_attr[0])
        assert len(ret_val.y) == 1 or len(ret_val.y) == num_nodes

        return ret_val


