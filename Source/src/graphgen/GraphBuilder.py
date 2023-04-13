import torch
import json
from torch_geometric.data import Data
from typing import Optional

class GraphBuilder:
    """Used to generate a graph instance of :obj:`torch_geometric.data.Data`"""

    def __init__(self, json_conf: Optional[str] = None) -> None:
        if (json_conf is not None):
            self._get_values_from_json(json_conf=json_conf)
        else:
            self._set_null_attributes()

    def generate_graph_from_json(self, json_path: str) -> Data:
        """Create a graph based on the contents of the JSON file"""
        json.load(open(json_path,'r'))
        pass

    def generate_random_graph(self) -> Data:
        return

    def _set_null_attributes(self) -> None:
        self._x = None
        self._edge_index = None
        self._edge_attr = None
        self._y = None

    def generate_edges(self, edge_count: int, feature_count: int) -> None:
        """Generates the graph connectivity matrix in COO format."""
        self._edge_index = torch.rand((2, edge_count), dtype=torch.long)
        self._edge_attr = torch.rand((edge_count, feature_count), dtype=torch.float)

    def generate_embeddings(self) -> None:
        self._x = torch.rand((self.node_count, self.node_attr_count), dtype=torch.float)

    def set_label(self, label: bool) -> None:
        """Set the global label of the graph (either True or False)"""
        self._y = label

    def generate_graph(self, path: str) -> Optional[Data]:
        """Generates the graph data into the directory specified by `path`."""
        if (self._x is not None) and (self._edge_index is not None) and (self._edge_attr is not None) and (self._y is not None):
            return Data(x=self._x, edge_index=self._edge_index, edge_attr=self._edge_attr, y=self._y)
        else:
            raise 


