import logging
import math

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator import GraphGenerator
import networkx as nx

log = logging.getLogger(__name__)
class SparseGraphBuilder(GraphGenerator):
    """Used to generate a graph instance of :obj:`torch_geometric.data.Data`."""

    def __init__(self, conf: dict, seed: int) -> None:
        r"""Receives a configuration and a randomizer seed to initialize the builder object. The configuration is a dictionary, with the following fields:
            {
                'directed': bool,
                'multiple_edges': bool,
                'loop_edges': bool,
                'node_count_lim': int,
                'node_feature_dim': int,
                EITHER
                    'density': float 0 < x < 1,
                OR
                    'edge_count': int
                'edge_feature_dim': int,
                'train_goal': 'node' or 'graph'
            }
        """
        required_fields = ['directed', 'multiple_edges', 'loop_edges', 'node_count_lim', 'node_feature_dim', 'edge_feature_dim', 'train_goal']
        assert conf is not None
        for field in required_fields:
            assert field in conf, f"The '{field}' field in the configuration of SparseGraphBuilder is not set!"

        assert ('density' in conf) ^ ('edge_count' in conf), "Exactly one of these two fields: 'density' or 'edge_count' should be set!"

        np.random.seed(seed=seed)
        self._conf: dict = conf

    def __call__(self) -> Data:
        """
        Overriden method of base class - generates a graph based on the configuration passed.
        """
        self._generate_graph_from_config()

        # couldn't get torch_geometric.utils.from_networkx() working
        edges = [(edge[0], edge[1]) for edge in self._G.edges]
        edge_index = torch.tensor(edges).t().contiguous()

        if self._conf['edge_feature_dim'] != 0:
            edge_attr = torch.tensor([self._G.edges[edge]['attr'] for edge in self._G.edges])
        else:
            edge_attr = None

        num_nodes = self._G.number_of_nodes()

        ret_val = Data(x=self._x, edge_index=edge_index, edge_attr=edge_attr, y=self._y)

        assert ret_val.x.shape[0] == num_nodes
        assert ret_val.x.shape[1] == len(self._G.nodes[0]['attr'])
        assert ret_val.edge_index.shape[0] == 2, f'edge_index attribute should have 2 columns, but has {ret_val.edge_index.shape[0]}\nedge_index: {ret_val.edge_index}'
        assert ret_val.edge_index.shape[1] == len(edges)
        if self._conf['edge_feature_dim'] != 0:
            assert ret_val.edge_attr.shape[0] == len(edges)
            assert ret_val.edge_attr.shape[1] == len(edge_attr[0])
        assert len(ret_val.y) == 1 or len(ret_val.y) == num_nodes

        return ret_val

    def _get_base_graph(self, directed: bool, multiple_edges: bool) -> None:
        """Assign to :obj:`self._G` one of the four main graph types"""
        types = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
        ret_ind = int(directed) + (2 * int(multiple_edges)) # binary indexing
        self._G = types[ret_ind]

    def _generate_graph_from_config(self) -> Data:
        """Create a graph based on the configuration received"""

        log.info(f"GraphBuilder received config:\n{self._conf}")
        self._get_base_graph(directed=self._conf['directed'], multiple_edges=self._conf['multiple_edges'])

        node_count = np.random.randint(30, self._conf['node_count_lim'])

        self._x = SparseGraphBuilder._generate_nodes(
            node_count=node_count, 
            node_feature_dim=self._conf['node_feature_dim'])
        
        self._add_nodes_from_list(node_feature_list=self._x)
        
        # If density is set, then it is used as default option to set the number of edges
        if 'density' in self._conf:
            self._conf['edge_count'] = math.floor(self._conf['density'] * ((node_count * (node_count - 1)) / 2))

        self._generate_edges(node_count=node_count, 
                             edge_count=self._conf['edge_count'], 
                             edge_feature_dim=self._conf['edge_feature_dim'], 
                             loop_edges=self._conf['loop_edges'], 
                             multiple_edges=self._conf['multiple_edges'])
        
        if self._conf['train_goal'] == 'node':
            self._generate_node_labels(x=self._x,
                                       node_count=node_count)
        elif self._conf['train_goal'] == 'graph':
            raise NotImplementedError('Graph labeling is not implemented yet, choose "node" option for now...')
        else:
            raise RuntimeError(f'Invalid configuration value for "train_goal" - {self._conf["train_goal"]}\nValid options: node, graph')
        
        return
        
    def _generate_nodes(node_count: int,
                        node_feature_dim: int) -> np.ndarray:
        r"""Randomly generate :obj:`node_feature_dim` features for each node in the graph and add them to the graph contained by the builder object."""
        x = np.random.rand(node_count, node_feature_dim) * 1000
        return x

    def _add_nodes_from_list(self, node_feature_list: np.ndarray) -> None:
        """Add nodes to the graph based on a 2D NumPy array supplied, where each row corresponds to the features of a single node in the graph."""
        for node_ind in range(node_feature_list.shape[0]):
            self._G.add_node(node_ind, attr=node_feature_list[node_ind].tolist())
        return

    def _generate_edges(self, node_count: int, edge_count: int, edge_feature_dim: int, loop_edges: bool, multiple_edges: bool) -> None:
        """Generate :obj:`edge_count` number of edges, each one with :obj:`edge_feature_dim` attributes."""
        added_edges = 0
        while added_edges != edge_count:
            # log.debug(f"Generating edges of graph [{added_edges}/{edge_count}]")
            generated_edge = tuple(np.random.choice(node_count, 2, replace=loop_edges))
            if not multiple_edges and self._G.has_edge(*generated_edge):
                continue
            if edge_feature_dim != 0:
                self._G.add_edge(
                    *generated_edge, 
                    attr=(np.random.rand(edge_feature_dim) * 10000).tolist())
            else:
                self._G.add_edge(*generated_edge)

            added_edges += 1
        return

    def _generate_node_labels(self, 
                              x: np.ndarray,
                              node_count: int) -> None:
        r"""Generates the node label for all nodes in the graph."""

        labels = []
        for node_index in range(node_count):
            label = SparseGraphBuilder._generate_node_label(
                vectors=x,
                node_index=node_index)
            labels.append(label)

        self._y = torch.tensor(labels)
        return

    def _generate_node_label(vectors: np.ndarray, node_index: int) -> float:
        r"""Generate the label of a node
        Args:
            :obj:`vectors` (required) - A 2D :obj:`ndarray` object of shape [<num_nodes>; <num_node_features>], 
        where each row corresponds to all features of one node.
            :obj:`node_index` (required) - The index of the node for which the last feature should be calcualted.
            
        Returns: The last feature of node at index :obj:`node_index`"""

        num_nodes = vectors.shape[0]
        num_dimensions = vectors.shape[1]

        # Extract the features of the current node
        node_features = vectors[node_index,:]

        # Compute the last feature based on the first three features of all nodes
        last_feature = 0.0
        for i in range(num_nodes):
            if i != node_index:
                node_i_features = vectors[i, :]
                for j in range(num_dimensions):
                    component = node_features[j] + node_i_features[j]
                    sined = np.sin(component)
                    last_feature += np.exp(sined)

        # Map the scaled value to one of the values 0, 1, 2, 3, or 4
        scaled_value = last_feature % 1
        mapped_value = int(scaled_value * 5)

        return mapped_value
        


