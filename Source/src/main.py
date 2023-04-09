import torch
from rewiring.model import GAT
from torch_geometric.datasets import Planetoid


# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')

# Initialize the model
model = GAT(dataset.num_features, 256, dataset.num_classes)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Define the loss function
criterion = torch.nn.NLLLoss()

# Train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
_, pred = model(dataset.x, dataset.edge_index).max(dim=1)
correct = float(pred[dataset.test_mask].eq(dataset.y[dataset.test_mask]).sum().item())
acc = correct / dataset.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))