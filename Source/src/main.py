import torch
import logging
import argparse
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.transform.DiffWire import DiffWire
from rewiring.PipelineController import PipelineController
from torch_geometric import datasets
from utils import get_rewiring_full_name


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

### Parse arguments
parser = argparse.ArgumentParser(description='Try out different rewiring techniques with different datasets and evaluate them.')
parser.add_argument('--model_type', required=True, type=str, choices=['GCN', 'GIN', 'GAT'], help='Options: GCN, GIN, GAT')
parser.add_argument('--rewiring', required=True, type=str, choices=['ARE', 'RD', 'DW'], help='Options: ARE = AddRandomEdges, RD = ReduceDiameter, DW = DiffWire')
parser.add_argument('--generate_plots', action='store_true', help='Set to output graph images to Source/out')
cfg = parser.parse_args()

print(f'Parsed arguments:\n  Model type:\t\t {cfg.model_type}\n  Rewiring strategy:\t {get_rewiring_full_name(cfg.rewiring)}\n  generate_plots:\t {cfg.generate_plots}')

### Define transformation
match cfg.rewiring:
    case 'ARE': rewiring = AddRandomEdges(ratio_edges=0.03)
    case 'RD': rewiring = ReduceDiameter()
    case 'DW': rewiring = DiffWire()
    case None: rewiring = None
    case _: raise ValueError(f'Invalid rewiring setting: {cfg.rewiring}')

### Load datasets
dset_base = "Source/res/datasets"
logger.info(f"Beginning to load datasets\tRoot: {dset_base}")
datasets = [
    # datasets.Planetoid(root=f'{dset_base}/cora', name='Cora'),
    # datasets.AmazonProducts(root=f'{dset_base}/amazonprods', transform=rewiring),
    # datasets.PPI(root=f'{dset_base}/ppi', transform=rewiring),
    # datasets.TUDataset(root=f'{dset_base}/enzymes', name='ENZYMES'),
    datasets.Planetoid(root=f'{dset_base}/cora_tf', name='Cora', transform=rewiring)
]
logger.info(f"Loading datasets DONE!")

controller = PipelineController(
    datasets=datasets,
    model_abbr=cfg.model_type,
    loss_function=torch.nn.NLLLoss()
)

controller.train_all()
controller.eval_all()