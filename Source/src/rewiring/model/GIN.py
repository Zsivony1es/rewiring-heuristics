import torch

import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, Dropout


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GIN, self).__init__()
        dim = 32
        self.conv1 = GINConv(Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim)))
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.conv2 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.conv3 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.conv4 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.conv5 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)