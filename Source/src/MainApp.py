import torch
import logging
import argparse
import time
from torch_geometric.transforms import GDC
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.pipeline.PipelineController import PipelineController
from rewiring.PlotGenerator import PlotGenerator
from rewiring.Utils import Utils


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

### Parse arguments
parser = argparse.ArgumentParser(description='Try out different rewiring techniques with different datasets and evaluate them.')
parser.add_argument('--dataset_list', required=True, type=str, help="""List of datasets which should be used for training, use '*' to 
                    train a model for every available dataset, or enter a comma-separated list of datasets, e.g. 'cora,enzymes'.
                    Make sure to add ' to the beginning and the end of this parameter no matter how you set it!
                    \nAvailable options: cora, citeseer, proteins, mutag, enzymes, lrgb-pascal, synthetic""")
parser.add_argument('--model_type', required=True, type=str, choices=['GCN', 'GAT'], help='Options: GCN, GAT')
parser.add_argument('--rewiring', required=False, type=str, choices=['ARE', 'RD', 'GDC'], help='Options: ARE = AddRandomEdges, RD = ReduceDiameter, GDC = Graph Diffusion Convolution')
parser.add_argument('--epochs', required=False, type=int, default=300, help='Set the amount of epochs each training cycle should include. Defaults to 300')
parser.add_argument('--generate_plots', action='store_true', help='Set to output graph images to Source/out')
cfg = parser.parse_args()

dataset_list = Utils.get_dataset_list(cfg.dataset_list)

print(f"""Parsed arguments:\n  Model type:\t\t {cfg.model_type}\n  Used datasets:\t {', '.join(dataset_list)}
  Rewiring strategy:\t {Utils.get_rewiring_full_name(cfg.rewiring)}\n  Epochs:\t\t {cfg.epochs}
  generate_plots:\t {cfg.generate_plots}""")

### Define transformation
rewiring_options = {
    'ARE': AddRandomEdges(min_density=0.2),
    'RD': ReduceDiameter(max_diam=25),
    'GDC' : GDC(),
    None: None
}
rewiring = rewiring_options.get(cfg.rewiring, ValueError(f'Invalid rewiring setting: {cfg.rewiring}'))

### Initialize pipeline
controller = PipelineController(
    datasets=dataset_list,
    model_abbr=cfg.model_type,
    rewiring=rewiring,
    loss_function=torch.nn.NLLLoss(),
    epochs = cfg.epochs,
    generate_plots = cfg.generate_plots,
)

print(f"Using device: {controller._device.type}")

accuracies = {}

for dataset_name in dataset_list:

  start = time.perf_counter()
  train_loss, val_loss, accuracy = controller.train_test_dataset(dataset_name)
  end = time.perf_counter()

  print(f"Elapsed time: {end-start:.4f}s")

  accuracies[dataset_name] = accuracy

  if cfg.generate_plots:
    pg = PlotGenerator()

    pg.generate_loss_plot(
            train_loss=train_loss,
            val_loss=val_loss,
            dset_name=dataset_name,
            model_type=cfg.model_type)
  
  Utils.delete_processed_folder(dataset_name)

    
pg.generate_accuracy_plot_multi(
      accuracy=accuracies,
      dset_name=dataset_name,
      model_type=cfg.model_type
    )
pg.leave_last_n_plots(n=10)