import torch
import torch_scatter
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1)
        self.fc_graph = torch.nn.Linear(hidden_channels, out_channels)
        self.fc_node = torch.nn.Linear(hidden_channels, out_channels)
        self.classification_task = None

    def set_classification_task(self, task):
        self.classification_task = task

    def forward(self, x, edge_index, batch = None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        if self.classification_task == 'graph':
            assert batch is not None, "'batch' parameter must be set during graph classification task!"
            x_graph = torch_scatter.scatter_max(x, batch, dim=0)[0]
            out = self.fc_graph(x_graph)
        elif self.classification_task == 'node':
            out = self.fc_node(x)
        else:
            raise ValueError("Invalid classification task. Please set the classification task using the set_classification_task() method.")
        
        return F.log_softmax(out, dim=1)