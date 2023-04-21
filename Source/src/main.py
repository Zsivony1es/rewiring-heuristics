import torch
import logging
from rewiring.model.GAT import GAT
from rewiring.transform.AddRandomEdges import AddRandomEdges
from torch_geometric import datasets

# TODO: Put this in a TestingDriver class

### Define transformation (AKA rewiring)
rewiring = AddRandomEdges(ratio_edges=0.03)

### Load datasets
dset_base = "Source/res"
logging.info(f"Beginning to load datasets\tRoot: {dset_base}")
datasets = [
    # datasets.Planetoid(root=f'{dset_base}/cora', name='Cora'),
    # datasets.AmazonProducts(root=f'{dset_base}/amazonprods', transform=rewiring),
    # datasets.PPI(root=f'{dset_base}/ppi', transform=rewiring),
    # datasets.TUDataset(root=f'{dset_base}/enzymes', name='ENZYMES'),
    datasets.Planetoid(root=f'{dset_base}/cora_tf', name='Cora', transform=rewiring)
]
logging.info(f"Loading datasets DONE!")

logging.info(f"Creating models with loaded datasets...")
# try GCN as well, GIN
models = list(map(lambda dset: GAT(dset.num_features, 256, dset.num_classes), datasets))
logging.info(f"Creating models DONE!")
optimizers = list(map(lambda model: torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4), models))

### Define the loss function
criterion = torch.nn.NLLLoss()

### Train the models
for i in range(len(datasets)):
    print(f"Beginning model training for {datasets[i].name}")
    models[i].train()
    for epoch in range(200):
        if (epoch + 1) % 25 == 0:
            print(f"Running epoch {epoch + 1}/200")
        optimizers[i].zero_grad()
        out = models[i](datasets[i].x, datasets[i].edge_index)
        loss = criterion(out[datasets[i].train_mask], datasets[i].y[datasets[i].train_mask])
        loss.backward()
        optimizers[i].step()
    ### Evaluate model
    models[i].eval()
    _, pred = models[i](datasets[i].x, datasets[i].edge_index).max(dim=1)
    correct = float(pred[datasets[i].test_mask].eq(datasets[i].y[datasets[i].test_mask]).sum().item())
    acc = correct / datasets[i].test_mask.sum().item()

    print(f'Accuracy ({datasets[i].name}): {acc}')