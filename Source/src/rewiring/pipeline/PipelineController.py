import torch
from torch import nn
import numpy as np
import logging
from typing import Any, Optional
from math import floor


from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from rewiring.pipeline.GraphClassificationPipeline import GraphClassificationPipeline
from rewiring.pipeline.SingleGraphNodeClassificationPipeline import SingleGraphNodeClassificationPipeline
from rewiring.pipeline.NodeClassificationPipeline import NodeClassifiactionPipeline
from rewiring.pipeline.DatasetManager import DatasetManager


from rewiring.model.GAT import GAT
from rewiring.model.GCN import GCN
from torch_geometric.nn import GIN

from rewiring.PlotGenerator import PlotGenerator
from rewiring.TrainValTestSplitter import TrainValTestSplitter


log = logging.getLogger(__name__)
class PipelineController():
    r"""This class is the main driver for training and evaluating models. For each dataset a different model will be created!
    Args:
        datasets (required) - List of all datasets on which the model should be trained and evaluated.
        model_abbr (required) - Set the GNN model to train. Options: [GCN, GAT, GIN]
        loss_function - Any loss function from PyTorch. (default: torch.nn.NLLLoss())
        """
    
    def __init__(self, 
                 datasets: list[str], 
                 model_abbr: str,
                 rewiring: Optional[BaseTransform], 
                 epochs: int, 
                 generate_plots, 
                 loss_function: Any = None) -> None:
        
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if generate_plots:
            self._plot_generator = PlotGenerator()
        else:
            self._plot_generator = None

        self._dataset_manager = DatasetManager(datasets, rewiring)
        self._model_used = model_abbr
        self._models = self._init_models(model_abbr)
        self._optimizers = self._init_optimizers()
        self._epochs = epochs

        if loss_function is None:
            if model_abbr == 'GIN':
                loss_function = nn.CrossEntropyLoss()
            else: 
                loss_function = nn.NLLLoss()

        self._loss_function = loss_function


    def _init_models(self, model_abbr: str) -> dict:
        log.info(f"Creating models with loaded datasets...")
        model_dict = {}
        for key, dset in self._dataset_manager.datasets.items():
            if model_abbr == 'GAT': 
                model_dict[key] = GAT(in_channels=dset.num_features, 
                                                  hidden_channels=128, 
                                                  out_channels=dset.num_classes)
            elif model_abbr == 'GIN': 
                model_dict[key] = GIN(in_channels=dset.num_features,
                                                  hidden_channels=16,
                                                  num_layers=2,
                                                  out_channels=dset.num_classes)
            elif model_abbr == 'GCN': 
                model_dict[key] = GCN(num_features=dset.num_features, 
                                                  num_classes=dset.num_classes)
            else:
                raise ValueError(f'Invalid model setting: {model_abbr}')
        log.info(f"Creating models DONE!")
        return model_dict
            
    def _init_optimizers(self) -> dict:
        opt = {}
        log.info(f"Creating optimizers...")
        log.debug(f"self._models: {self._models}")
        for key, model in self._models.items():
            opt[key] = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        log.info(f"Creating optimizers DONE!")
        return opt
        
    def train_test_dataset(self, dataset_name: str) -> tuple:
        r"""Train the model corresponding to the specified dataset.
        Args:
            :param:`dataset_name` (required) - The name of the dataset for which the corresponding model should be trained.
        Returns: 
            A tuple of two lists containing the loss measured in each epoch on the training and the validation set respectively, 
            so with 100 epochs, each list should be 100 in length."""
        
        # Get model and optimizer
        model = self._models[dataset_name]
        optimizer = self._optimizers[dataset_name]

        # Get dataset and the corresponding loaders
        dataset = self._dataset_manager.datasets[dataset_name]
        
        # Get pipelines set up
        if dataset_name.lower() in ['cora', 'citeseer']:

            model.set_classification_task(task='node')

            pipeline = SingleGraphNodeClassificationPipeline(
                device=self._device,
                model=model,
                criterion=self._loss_function,
                optimizer=optimizer
            )

            pipeline.set_masks(data=dataset)
            
        elif dataset_name.lower() in ['mutag', 'enzymes', 'proteins']:

            train_loader = self._dataset_manager.train_loader[dataset_name]
            val_loader = self._dataset_manager.val_loader[dataset_name]
            test_loader = self._dataset_manager.test_loader[dataset_name]

            model.set_classification_task(task='graph')

            pipeline = GraphClassificationPipeline(
                device=self._device,
                model=model,
                criterion=self._loss_function,
                optimizer=optimizer
            )
        elif dataset_name.lower() in ['lrgb-pascal', 'synthetic']:

            train_loader = self._dataset_manager.train_loader[dataset_name]
            val_loader = self._dataset_manager.val_loader[dataset_name]
            test_loader = self._dataset_manager.test_loader[dataset_name]

            model.set_classification_task(task='node')

            pipeline = NodeClassifiactionPipeline(
                device=self._device,
                model=model,
                criterion=self._loss_function,
                optimizer=optimizer
            )

        else:
            raise RuntimeError(f"Unknown dataset name: {dataset_name}")
        
        # Begin training
        if dataset_name.lower() in ['cora', 'citeseer']:
            train_losses, val_losses, accuracy_list = pipeline.train_validate(
                data=dataset,
                epochs=self._epochs,
                log=True
            )
        else:
            train_losses, val_losses, accuracy_list = pipeline.train_validate(
                train_data=train_loader,
                val_data=val_loader,
                epochs=self._epochs,
                log=True
            )

        # Extract trained model
        self._models[dataset_name] = pipeline.model

        # Evaluate on test dataset
        if dataset_name.lower() in ['cora', 'citeseer']:
            loss, acc = pipeline.single_validate(self._dataset_manager.datasets[dataset_name])
        else:
            loss, acc = pipeline.single_validate(test_loader)
        print(f"==== Results on test set ====\nLoss: {loss}\tAccuracy: {acc}")

        return (train_losses, val_losses, accuracy_list)
    
    def train_test_all(self):
        pass