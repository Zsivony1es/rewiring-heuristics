import torch
import logging
import argparse
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.PipelineController import PipelineController
from utils import Utils



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

### Parse arguments
parser = argparse.ArgumentParser(description='Try out different rewiring techniques with different datasets and evaluate them.')
parser.add_argument('--dataset_list', required=True, type=str, help="""List of datasets which should be used for training, use * to 
                    train a model for every available dataset, or enter a comma-separated list of datasets, e.g. cora,enzymes.
                    \nAvailable options: cora, enzymes, amazon, ppi""")
parser.add_argument('--model_type', required=True, type=str, choices=['GCN', 'GIN', 'GAT'], help='Options: GCN, GIN, GAT')
parser.add_argument('--rewiring', required=False, type=str, choices=['ARE', 'RD', 'DW'], help='Options: ARE = AddRandomEdges, RD = ReduceDiameter, DW = DiffWire')
parser.add_argument('--generate_plots', action='store_true', help='Set to output graph images to Source/out')
cfg = parser.parse_args()

util = Utils()
print(f'Parsed arguments:\n  Model type:\t\t {cfg.model_type}\n  Used datasets:\t {cfg.dataset_list}\n  Rewiring strategy:\t {util.get_rewiring_full_name(cfg.rewiring)}\n  generate_plots:\t {cfg.generate_plots}')

### Define transformation
rewiring_options = {
    'ARE': AddRandomEdges(),
    'RD': ReduceDiameter(max_diam=3),
    None: None
}

rewiring = rewiring_options.get(cfg.rewiring, ValueError(f'Invalid rewiring setting: {cfg.rewiring}'))

### Load datasets
datasets = util.get_dataset_dictionary(dset_conf=cfg.dataset_list, rewiring=rewiring)

### Initialize pipeline
controller = PipelineController(
    datasets=datasets,
    model_abbr=cfg.model_type,
    loss_function=torch.nn.NLLLoss()
)

controller.train_all()
controller.eval_all()