import torch
import logging
import argparse

from rewiring.model.GAT import GAT
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.transform.DiffWire import DiffWire
from torch_geometric import datasets
from utils import get_rewiring_full_name

# TODO: Put this in a TestingDriver class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Parse arguments
parser = argparse.ArgumentParser(description='Try out different rewiring techniques with different datasets and evaluate them.')
parser.add_argument('--model_type', required=True, type=str, choices=['GCN', 'GIN', 'GAT'], help='Options: GCN, GIN, GAT')
parser.add_argument('--rewiring', required=True, type=str, choices=['ARE', 'RD', 'DW'], help='Options: ARE = AddRandomEdges, RD = ReduceDiameter, DW = DiffWire')
parser.add_argument('--generate_plots', action='store_true', help='Set to output graph images to Source/out')
cfg = parser.parse_args()

logger.info(f'Parsed arguments:\tModel type: {cfg.model_type}\t\tRewiring strategy: {get_rewiring_full_name(cfg.rewiring)}\t\tgenerate_plots: {cfg.generate_plots}')

### Define transformation (AKA rewiring)
match cfg.rewiring:
    case 'ARE': rewiring = AddRandomEdges(ratio_edges=0.03)
    case 'RD': rewiring = ReduceDiameter()
    case 'DW': rewiring = DiffWire()
    case _: raise ValueError(f'Invalid rewiring setting: {abbr}')

### Load datasets
dset_base = "Source/res"
logger.info(f"Beginning to load datasets\tRoot: {dset_base}")
datasets = [
    # datasets.Planetoid(root=f'{dset_base}/cora', name='Cora'),
    # datasets.AmazonProducts(root=f'{dset_base}/amazonprods', transform=rewiring),
    # datasets.PPI(root=f'{dset_base}/ppi', transform=rewiring),
    # datasets.TUDataset(root=f'{dset_base}/enzymes', name='ENZYMES'),
    datasets.Planetoid(root=f'{dset_base}/cora_tf', name='Cora', transform=rewiring)
]
logger.info(f"Loading datasets DONE!")

logger.info(f"Creating models with loaded datasets...")
# try GCN as well, GIN
models = list(map(lambda dset: GAT(dset.num_features, 256, dset.num_classes), datasets))

logger.info(f"Creating models DONE!")
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