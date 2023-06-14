import torch
from torch_geometric.nn import GCNConv
from torch.nn import Module


class SimpleGCN(Module):
    """
    I wasn't able to get multiple graph node classification working with the other GCN model quickly enough,
    so to save time, I'm temporarily using this model.
    """
    def __init__(self, num_features, num_classes, hidden_size = 16):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)
