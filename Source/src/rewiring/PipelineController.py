import torch
import logging
import argparse
from typing import Any

from rewiring.model.GAT import GAT
from rewiring.model.GCN import GCN
from rewiring.model.GIN import GIN

from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.transform.DiffWire import DiffWire
from rewiring.PlotGenerator import PlotGenerator
from torch_geometric import datasets
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.transforms import BaseTransform
from utils import get_rewiring_full_name

class PipelineController():

    def __init__(self, datasets: list[Dataset | InMemoryDataset], model_abbr: str, loss_function: Any = torch.nn.NLLLoss()) -> None:
        self._logger = logging.getLogger(__name__)
        self._plot_generator = PlotGenerator()
        self._datasets = datasets
        self._models = self._init_models(model_abbr)
        self._optimizers = self._init_optimizers()
        self._loss_function = loss_function
        pass

    def _init_models(self, model_abbr: str) -> list[torch.nn.Model]:
        self._logger.info(f"Creating models with loaded datasets...")
        model_list = None
        match model_abbr:
            case 'GAT': model_list = list(map(lambda dset: GAT(dset.num_features, 256, dset.num_classes), self._datasets))
            case 'GIN': model_list = list(map(lambda dset: GIN(dset.num_features, 256, dset.num_classes), self._datasets))
            case 'GCN': model_list = list(map(lambda dset: GCN(dset.num_features, 256, dset.num_classes), self._datasets))
            case _: raise ValueError(f'Invalid model setting: {model_abbr}')
        if model_list is not None:
            self._logger.info(f"Creating models DONE!")
            
    def _init_optimizers(self) -> list[torch.optim.Adam]:
        opt = None
        self._logger.info(f"Creating optimizers...")
        opt = list(map(lambda model: torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4), self._models))
        if opt is not None:
            self._logger.info(f"Creating optimizers DONE!")
            return opt

    def train(self, dataset_index: int) -> None:
        print(f"Beginning model training for {self._datasets[dataset_index].name}")
        self._models[dataset_index].train()
        for epoch in range(200):
            if (epoch + 1) % 25 == 0:
                print(f"Running epoch {epoch + 1}/200")
            self._optimizers[dataset_index].zero_grad()
            out = self._models[dataset_index](self._datasets[dataset_index].x, self._datasets[dataset_index].edge_index)
            loss = self._loss_function( out[self._datasets[dataset_index].train_mask], self._datasets[dataset_index].y[self._datasets[dataset_index].train_mask])
            loss.backward()
            self._optimizers[dataset_index].step()
        print(f"Model training of {self._datasets[dataset_index].name} finished!")

    def eval(self, dataset_index: int) -> None:
        self._models[dataset_index].eval()
        _, pred = self._models[dataset_index](self._datasets[dataset_index].x, self._datasets[dataset_index].edge_index).max(dim=1)
        correct = float(pred[self._datasets[dataset_index].test_mask].eq(self._datasets[dataset_index].y[self._datasets[dataset_index].test_mask]).sum().item())
        acc = correct / self._datasets[dataset_index].test_mask.sum().item()

        print(f'Accuracy ({self._datasets[dataset_index].name}): {acc}')

    def train_all(self) -> None:
        for dataset_index in range(len(self._datasets)):
            self.train(dataset_index)

    def eval_all(self) -> None:
        for dataset_index in range(len(self._datasets)):
            self.eval(dataset_index)