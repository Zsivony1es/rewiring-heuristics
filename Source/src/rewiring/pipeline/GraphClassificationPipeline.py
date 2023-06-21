import logging
import torch
import torch_geometric
import numpy as np
from typing import Union

from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from rewiring.pipeline.AbstractTrainingPipeline import AbstractTrainingPipeline


log = logging.getLogger(__name__)
class GraphClassificationPipeline(AbstractTrainingPipeline):
    def __init__(self,
                device: torch.device,
                model: Module,
                criterion,
                optimizer) -> None:
        """
        Creates a pipeline designed to execute model training and evaluation for graph classification tasks.
        """
        super().__init__(device=device,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer)
        

    def single_train(self,
                     train_data: Union[Dataset, DataLoader]) -> float:
        """
        Trains the model on a given dataset for a single epoch. Returns the training loss.
        """
        epoch_train_loss = []

        self.model.train()
        for batch in train_data:
            # Clear the gradients
            self.optimizer.zero_grad()
            # Forward Pass
            out = self.model(batch.x.to(self.device), batch.edge_index.to(self.device), batch.batch.to(self.device))
            # Find the Loss
            train_loss = self.criterion(out, batch.y)
            epoch_train_loss.append(train_loss.item())    
            # Calculate gradients
            train_loss.backward()
            # Update Weights
            self.optimizer.step()

        avg_loss = np.array(epoch_train_loss).mean().item()
        return avg_loss

    def single_validate(self,
                 val_data: Union[Dataset, DataLoader]) -> tuple:
        """
        Performs validation on the given dataset for a single epoch. Returns the validation loss and the accuracy.
        """
        correct = 0
        epoch_val_loss = []

        self.model.eval()
        
        for batch in val_data:
            with torch.no_grad():
                out = self.model(batch.x.to(self.device), batch.edge_index.to(self.device), batch.batch.to(self.device))
                val_loss = self.criterion(out, batch.y)
                epoch_val_loss.append(val_loss.item())
                _, pred = torch.max(out.data, 1)
                correct += (pred == batch.y).float().sum().item()
        
        avg_loss = np.array(epoch_val_loss).mean().item()
        acc = correct / len(val_data.dataset)

        return (avg_loss, acc)

    def train_validate(self,
                       train_data: Union[Dataset, DataLoader],
                       val_data: Union[Dataset, DataLoader],
                       epochs: int,
                       log: bool = False) -> tuple:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively as a tuple. If log is set to True, it will display
        the progress.
        """
        return super().train_validate(train_data=train_data,
                                      val_data=val_data,
                                      epochs=epochs,
                                      log=log)
