import torch
import logging
from typing import Any

from rewiring.model.GAT import GAT
from rewiring.model.GCN import GCN
from rewiring.model.GIN import GIN

from rewiring.PlotGenerator import PlotGenerator


class PipelineController():
    r"""This class is the main driver for training and evaluating models. For each dataset a different model will be created!
    Args:
        datasets (required) - List of all datasets on which the model should be trained and evaluated.
        model_abbr (required) - Set the GNN model to train. Options: [GCN, GAT, GIN]
        loss_function - Any loss function from PyTorch. (default: torch.nn.NLLLoss())
        """
    
    def __init__(self, datasets: dict, model_abbr: str, loss_function: Any = torch.nn.NLLLoss()) -> None:
        self._logger = logging.getLogger(__name__)
        self._plot_generator = PlotGenerator()
        self._datasets = datasets
        self._models = self._init_models(model_abbr)
        self._optimizers = self._init_optimizers()
        self._loss_function = loss_function
        pass

    def _init_models(self, model_abbr: str) -> list[torch.nn.Module]:
        self._logger.info(f"Creating models with loaded datasets...")
        model_dict = {}
        for key, dset in self._datasets.items():
            match model_abbr:
                case 'GAT': model_dict[key] = GAT(dset.num_features, 256, dset.num_classes)
                case 'GIN': model_dict[key] = GIN(dset.num_features, dset.num_classes)
                case 'GCN': model_dict[key] = GCN(dset.num_features, dset.num_classes)
                case _: raise ValueError(f'Invalid model setting: {model_abbr}')
        self._logger.info(f"Creating models DONE!")
        return model_dict
            
    def _init_optimizers(self) -> list[torch.optim.Adam]:
        opt = {}
        self._logger.info(f"Creating optimizers...")
        self._logger.debug(f"self._models: {self._models}")
        for key, model in self._models.items():
            opt[key] = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        self._logger.info(f"Creating optimizers DONE!")
        return opt


    def train(self, dataset_name: str) -> None:
        r"""Train the model corresponding to the specified dataset.
        Args:
            dataset_name (required) - The name of the dataset for which a model should be trained."""
        
        used_dataset = self._datasets[dataset_name]
        print(f"Beginning model training for {used_dataset.name}")
        self._models[dataset_name].train()

        for epoch in range(1, 201):
            if epoch % 25 == 0:
                print(f"Running epoch {epoch}/200")

            self._optimizers[dataset_name].zero_grad()

            out = self._models[dataset_name](
                used_dataset.x,
                used_dataset.edge_index)
            
            loss = self._loss_function( 
                out[used_dataset.train_mask], 
                used_dataset.y[used_dataset.train_mask])
            
            loss.backward()
            self._optimizers[dataset_name].step()
        print(f"Model training of {used_dataset.name} finished!")

    def eval(self, dataset_name: str) -> None:
        r"""Evaluate the model corresponding to the specified dataset.
        Args:
            dataset_name (required) - The name of the dataset for which a model should be evaluated."""
        
        model = self._models[dataset_name]
        dset = self._datasets[dataset_name]
        
        self._models[dataset_name].eval()
        _, pred = model(dset.x, dset.edge_index).max(dim=1)
        correct = float(pred[dset.test_mask].eq(dset.y[dset.test_mask]).sum().item())
        acc = correct / dset.test_mask.sum().item()

        print(f'Accuracy ({dset.name}): {acc}')

    def train_all(self) -> None:
        r"""Train the models with all datasets."""
        for key, _ in self._datasets.items():
            self.train(key)

    def eval_all(self) -> None:
        r"""Evaluate the models on all datasets."""
        for key, _ in self._datasets.items():
            self.eval(key)