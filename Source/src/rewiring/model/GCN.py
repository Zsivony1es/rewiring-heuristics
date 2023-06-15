import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.fc_graph = nn.Linear(hidden_size, num_classes)
        self.fc_node = nn.Linear(hidden_size, num_classes)
        self.classification_task = None

    def set_classification_task(self, task):
        self.classification_task = task

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch = None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        if self.classification_task == 'graph':
            assert batch is not None, "'batch' parameter must be set during graph classification task!"
            x_graph = torch_scatter.scatter_max(x, batch, dim=0)[0]
            out = self.fc_graph(x_graph)
        elif self.classification_task == 'node':
            out = self.fc_node(x)
        else:
            raise ValueError("Invalid classification task. Please set the classification task using the set_classification_task() method.")
        
        return F.log_softmax(out, dim=1)