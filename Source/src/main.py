import torch
from rewiring.structures.GAT import GAT
from rewiring.transform.AddRandomEdge import AddRandomEdge
from torch_geometric import datasets

# Define transformation (AKA rewiring)

rewiring = AddRandomEdge()

# Load datasets
dset_base = "Source/res"
datasets = [
    datasets.Planetoid(root=f'{dset_base}/cora', name='Cora', transform=rewiring),
    # datasets.AmazonProducts(root=f'{dset_base}/amazonprods', transform=rewiring),
    # datasets.PPI(root=f'{dset_base}/ppi', transform=rewiring),
    datasets.Planetoid(root=f'{dset_base}/reddit', name='CiteSeer', transform=rewiring),
    datasets.TUDataset(root=f'{dset_base}/enzymes', name='ENZYMES',)
]

models = list(map(lambda dset: GAT(dset.num_features, 256, dset.num_classes), datasets))
optimizers = list(map(lambda model: torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4), models))

# Define the loss function
criterion = torch.nn.NLLLoss()

# Train the models
for i in range(len(datasets)):

    models[i].train()
    for epoch in range(200):
        print(f"Running epoch {epoch}/200")
        optimizers[i].zero_grad()
        out = models[i](datasets[i].x, datasets[i].edge_index)
        loss = criterion(out[datasets[i].train_mask], datasets[i].y[datasets[i].train_mask])
        loss.backward()
        optimizers[i].step()
    # Evaluate model
    models[i].eval()
    _, pred = models[i](datasets[i].x, datasets[i].edge_index).max(dim=1)
    correct = float(pred[datasets[i].test_mask].eq(datasets[i].y[datasets[i].test_mask]).sum().item())
    acc = correct / datasets[i].test_mask.sum().item()

    print(f'Accuracy ({datasets[i].name}): {acc}')