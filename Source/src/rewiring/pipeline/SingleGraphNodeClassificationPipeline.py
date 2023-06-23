import logging
import torch
import numpy as np
from typing import Union

from torch.nn import Module
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from rewiring.pipeline.AbstractTrainingPipeline import AbstractTrainingPipeline
from rewiring.TrainValTestSplitter import TrainValTestSplitter

log = logging.getLogger(__name__)
class SingleGraphNodeClassificationPipeline(AbstractTrainingPipeline):
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

        assert self.train_mask is not None, "Training mask is not set! Please call :obj:`set_masks()` before executing training!"

        self.model.to(self.device)
        self.model.train()
        # Clear the gradients
        self.optimizer.zero_grad()
        # Forward Pass
        out = self.model(train_data.x.to(self.device), train_data.edge_index.to(self.device))
        # Find the Loss
        train_loss = self.criterion(out[self.train_mask].to(self.device), train_data.y[self.train_mask].to(self.device))    
        # Calculate gradients
        train_loss.backward()
        # Update Weights
        self.optimizer.step()

        return train_loss.item()

    def single_validate(self,
                        val_data: Union[Dataset, DataLoader]) -> tuple:
        """
        Performs validation on the given dataset for a single epoch. Returns the validation loss and the accuracy.
        """
        assert self.val_mask is not None, "Validation mask is not set! Please call :obj:`set_masks()` before executing training!"

        self.model.eval()
        with torch.no_grad():
            out = self.model(val_data.x.to(self.device), val_data.edge_index.to(self.device))
            val_loss = self.criterion(out[self.val_mask].to(self.device), val_data.y[self.val_mask].to(self.device))
            _, pred = torch.max(out.data.to(self.device), 1)
            correct = (pred[self.val_mask].to(self.device) == val_data.y[self.val_mask].to(self.device)).float().sum().item()
            acc = correct / self.val_mask.shape[0]

        return (val_loss.item(), acc)

    def train_validate(self,
                       data: Union[Dataset, DataLoader],
                       epochs: int,
                       log: bool = False) -> tuple:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively as a tuple. If log is set to True, it will display
        the progress.
        """

        return super().train_validate(train_data=data,
                                      val_data=data,
                                      epochs=epochs,
                                      log=log)
    
    def set_masks(self,
                  data: Union[Dataset, InMemoryDataset],
                  val_ratio: float = 0.2,
                  test_ratio: float = 1) -> None:
        """
        Creates the training, validation and test masks required for the pipeline to work.
        """
        assert val_ratio >= 0 and val_ratio < 1, "The ratio of the validation set to the whole dataset must be in [0,1)!"
        assert test_ratio > 0 and test_ratio <= 1, "The ratio of the test set to the whole dataset must be in (0,1]!"

        self.train_mask, self.val_mask, self.test_mask = \
            TrainValTestSplitter._single_graph_split(data, 
                                                    val_size=val_ratio, 
                                                    test_size=test_ratio)