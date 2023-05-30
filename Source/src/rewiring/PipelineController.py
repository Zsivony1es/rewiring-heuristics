import torch
import numpy as np
import logging
from typing import Any
from math import floor

from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

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
    
    def __init__(self, datasets: dict, model_abbr: str, epochs: int, generate_plots, loss_function: Any = torch.nn.NLLLoss()) -> None:
        self._logger = logging.getLogger(__name__)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if generate_plots:
            self._plot_generator = PlotGenerator()
        else:
            self._plot_generator = None
        self._datasets = datasets
        self._model_used = model_abbr
        self._models = self._init_models(model_abbr)
        self._optimizers = self._init_optimizers()
        self._epochs = epochs
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


    def train(self, dataset_name: str) -> tuple[list[float], list[float]]:
        r"""Train the model corresponding to the specified dataset.
        Args:
            dataset_name (required) - The name of the dataset for which the corresponding model should be trained.
        Returns: 
            A tuple of two lists containing the loss measured in each epoch on the training and the validation set respectively, 
            so with 100 epochs, each list should be 100 in length."""
        
        dset: Dataset | InMemoryDataset = self._datasets[dataset_name]
        model = self._models[dataset_name]
        print(f"Beginning model training for {dset.name}")

        # Divide the dataset into training and validation sets
        training_graph_count = floor(0.85 * len(dset))
        all_indices = np.arange(len(dset))
        training_indices = np.random.choice(all_indices, size=training_graph_count, replace=False)
        validation_indices = np.setdiff1d(all_indices, training_indices)

        train_dset = dset.index_select(training_indices)
        val_dset = dset.index_select(validation_indices)

        train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dset, batch_size=64, shuffle=False)

        train_losses = []
        val_losses = []

        for epoch in range(1, self._epochs + 1):
            ### TRAINING   
            model.train()
            step_loss = []
            for batch in train_loader:
                data = batch.to(self._device)     
                # Clear the gradients
                self._optimizers[dataset_name].zero_grad()

                # Forward Pass
                out = model(data)
                
                # Find the Loss
                train_loss = self._loss_function( 
                    out[train_dset.train_mask], 
                    train_dset.y[train_dset.train_mask])
                step_loss.append(val_loss.item())

                # Calculate gradients
                train_loss.backward()
                
                # Update Weights
                self._optimizers[dataset_name].step()

            train_losses.append(np.array(step_loss).mean())

            ### VALIDATION
            model.eval()
            step_loss = []
            with torch.no_grad():
                for batch in val_loader:
                    data = batch.to(self._device)
                    # Forward pass
                    out = model(data)
                    # Find the loss
                    val_loss = self._loss_function(
                        out[val_dset.train_mask], 
                        val_dset.y[val_dset.train_mask])
                    step_loss.append(val_loss.item())

            val_losses.append(np.array(step_loss).mean())

            if epoch % 25 == 0:
                print(f'Epoch [{epoch}/{self._epochs}]\t Loss: {val_loss.item():.4f}')

        print(f"Model training of {dset.name} finished!")
        return (train_losses, val_losses)


    def eval(self, dataset_name: str) -> None:
        r"""Evaluate the model corresponding to the specified dataset.
        Args:
            dataset_name (required) - The name of the dataset for which the corresponding model should be evaluated."""
        
        model = self._models[dataset_name]
        dset = self._datasets[dataset_name]
        
        model.eval()
        _, pred = model(dset.x, dset.edge_index).max(dim=1)
        correct = float(pred[dset.test_mask].eq(dset.y[dset.test_mask]).sum().item())
        acc = correct / dset.test_mask.sum().item()

        print(f'Accuracy ({dset.name}): {acc}')

    def train_all(self) -> None:
        r"""Train the models with all datasets."""
        for key, _ in self._datasets.items():
            (train_loss, val_loss) = self.train(key)
            if self._plot_generator is not None:
                self._plot_generator.generate_loss_plot(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    dset_name=key.title(),
                    model_type=self._model_used)
                
    def eval_all(self) -> None:
        r"""Evaluate the models on all datasets."""
        for key, _ in self._datasets.items():
            self.eval(key)