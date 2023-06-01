import torch
import logging
import argparse
from torch_geometric.transforms import GDC
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.PipelineController import PipelineController
from utils import Utils


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

### Parse arguments
parser = argparse.ArgumentParser(description='Try out different rewiring techniques with different datasets and evaluate them.')
parser.add_argument('--dataset_list', required=True, type=str, help="""List of datasets which should be used for training, use * to 
                    train a model for every available dataset, or enter a comma-separated list of datasets, e.g. cora,enzymes.
                    \nAvailable options: cora, enzymes, amazon, ppi""")
parser.add_argument('--model_type', required=True, type=str, choices=['GCN', 'GIN', 'GAT'], help='Options: GCN, GIN, GAT')
parser.add_argument('--rewiring', required=False, type=str, choices=['ARE', 'RD', 'GDC'], help='Options: ARE = AddRandomEdges, RD = ReduceDiameter, GDC = Graph Diffusion Convolution')
parser.add_argument('--epochs', required=False, type=int, default=300, help='Set the amount of epochs each training cycle should include. Defaults to 300')
parser.add_argument('--generate_plots', action='store_true', help='Set to output graph images to Source/out')
cfg = parser.parse_args()

util = Utils()
print(f"""Parsed arguments:\n  Model type:\t\t {cfg.model_type}\n  Used datasets:\t {cfg.dataset_list}
  Rewiring strategy:\t {util.get_rewiring_full_name(cfg.rewiring)}\n  Epochs:\t\t {cfg.epochs}
  generate_plots:\t {cfg.generate_plots}""")

### Define transformation
rewiring_options = {
    'ARE': AddRandomEdges(),
    'RD': ReduceDiameter(max_diam=3),
    'GDC' : GDC(),
    None: None
}

rewiring = rewiring_options.get(cfg.rewiring, ValueError(f'Invalid rewiring setting: {cfg.rewiring}'))

### Load datasets
datasets = util.get_dataset_dictionary(dset_conf=cfg.dataset_list, rewiring=rewiring)

### Initialize pipeline
controller = PipelineController(
    datasets=datasets,
    model_abbr=cfg.model_type,
    loss_function=torch.nn.NLLLoss(),
    epochs = cfg.epochs,
    generate_plots = cfg.generate_plots,
)

train_loss, val_loss = controller.train_test_dataset('cora')

if controller._plot_generator is not None:
    controller._plot_generator.generate_loss_plot(
        train_loss=train_loss,
        val_loss=val_loss,
        dset_name='Cora',
        model_type=cfg.model_type)