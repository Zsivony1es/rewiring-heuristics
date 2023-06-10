import torch
from torch import nn
import numpy as np
import logging
from typing import Any
from math import floor

from torch_geometric.data import Dataset, InMemoryDataset

from rewiring.model.GAT import GAT
from rewiring.model.GCN import GCN
from rewiring.model.GIN import GIN

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
    
    def __init__(self, datasets: dict, model_abbr: str, epochs: int, generate_plots, loss_function: Any = torch.nn.NLLLoss()) -> None:
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
        log.info(f"Creating models with loaded datasets...")
        model_dict = {}
        for key, dset in self._datasets.items():
            match model_abbr:
                case 'GAT': model_dict[key] = GAT(dset.num_features, 256, dset.num_classes)
                case 'GIN': model_dict[key] = GIN(dset.num_features, dset.num_classes)
                case 'GCN': model_dict[key] = GCN(dset.num_features, dset.num_classes)
                case _: raise ValueError(f'Invalid model setting: {model_abbr}')
        log.info(f"Creating models DONE!")
        return model_dict
            
    def _init_optimizers(self) -> list[torch.optim.Adam]:
        opt = {}
        log.info(f"Creating optimizers...")
        log.debug(f"self._models: {self._models}")
        for key, model in self._models.items():
            opt[key] = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        log.info(f"Creating optimizers DONE!")
        return opt

    def _train(model: nn.Module, 
              data: Dataset | InMemoryDataset, 
              optimizer, 
              criterion, 
              num_epochs: int,
              train_mask: torch.Tensor, 
              val_mask: torch.Tensor = None) -> tuple[nn.Module, list[float], list[float]]:
        """Trains a model on a given dataset, with a train mask and an optional validation mask.
        Returns a tuple of the trained model, a list of the training and validation losses."""
        
        train_losses = []
        val_losses = []
        acc_list = []


        print(f"Beginning model training...")    

        for epoch in range(num_epochs):
            
            ### TRAINING
            model.train()
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            out = model(data.x, data.edge_index)
            # Find the Loss
            train_loss = criterion(out[train_mask], data.y[train_mask])
            train_losses.append(train_loss.item())
            
            # Calculate gradients
            train_loss.backward()

            # Update Weights
            optimizer.step()

            if val_mask is None:
                if (epoch + 1) % 10 == 0:
                    print(f"""Epoch: [{epoch+1}/{num_epochs}]
                      Training loss: {train_loss.item():.4f}""")
                continue

            ### VALIDATION
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = criterion(out[val_mask], data.y[val_mask])
                val_losses.append(val_loss.item())
                _, pred = torch.max(out.data, 1)
                correct = (pred[val_mask] == data.y[val_mask]).float().sum().item()
                acc = correct / val_mask.shape[0]
                acc_list.append(acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"""Epoch: [{epoch+1}/{num_epochs}]
                  Training loss: {train_loss.item():.4f},\t Validation loss: {val_loss.item():.4f}
                  Accuracy: {acc}""")

        return (model, train_losses, val_losses, acc_list)
        
    def train_test_dataset(self, dataset_name: str) -> tuple[list[float], list[float]]:
        r"""Train the model corresponding to the specified dataset.
        Args:
            :param:`dataset_name` (required) - The name of the dataset for which the corresponding model should be trained.
        Returns: 
            A tuple of two lists containing the loss measured in each epoch on the training and the validation set respectively, 
            so with 100 epochs, each list should be 100 in length."""
        
        dset: Dataset | InMemoryDataset = self._datasets[dataset_name]
        if dataset_name == 'cora':
            dset = dset[0]
        model = self._models[dataset_name]
        optimizer = self._optimizers[dataset_name]
        print(f"Beginning model training for {dataset_name}")

        # Divide the dataset into training, validation and test sets
        train_mask, val_mask, test_mask = TrainValTestSplitter.split(dataset_name=dataset_name, dataset=dset)

        trained_model, train_loss, val_loss, accuracy_list = PipelineController._train(
            model=model,
            data=dset,
            optimizer=optimizer,
            criterion=self._loss_function,
            num_epochs=self._epochs,
            train_mask=train_mask,
            val_mask=val_mask)
        
        self._models[dataset_name] = trained_model
        model = trained_model

        print(f"Model training for {dataset_name} finished!")
        print(f"Evaluating {dataset_name} model on test set...")
        loss, acc = PipelineController._evaluate(
            model=trained_model, 
            data=dset, 
            test_mask=test_mask, 
            criterion=self._loss_function)
        
        print(f"""  Test loss: {loss:.4f}
          Accuracy: {acc}""")

        return (train_loss, val_loss, accuracy_list)
    
    def _evaluate(model: nn.Module, 
                  data: Dataset | InMemoryDataset, 
                  test_mask: torch.Tensor,
                  criterion):
        
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            loss = criterion(out[test_mask], data.y[test_mask])
            _, pred = out.max(dim=1)
            correct = (pred[test_mask] == data.y[test_mask]).float().sum().item()
            acc = correct / test_mask.shape[0]
        
        return (loss.item(), acc) 
    

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
            (train_loss, val_loss) = self.train_dataset(dataset_name=key)
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