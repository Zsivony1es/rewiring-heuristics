import logging
import torch
import numpy as np

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
                     train_data: Dataset | DataLoader) -> float:
        """
        Trains the model on a given dataset for a single epoch. Returns the training loss.
        """

        assert self.train_mask is not None, "Training mask is not set! Please call :obj:`set_masks()` before executing training!"

        self.model.train()
        # Clear the gradients
        self.optimizer.zero_grad()
        # Forward Pass
        out = self.model(train_data.x.to(self.device), train_data.edge_index.to(self.device))
        # Find the Loss
        train_loss = self.criterion(out[self.train_mask], train_data.y[self.train_mask])    
        # Calculate gradients
        train_loss.backward()
        # Update Weights
        self.optimizer.step()

    def single_validate(self,
                        val_data: Dataset | DataLoader) -> tuple[float, float]:
        """
        Performs validation on the given dataset for a single epoch. Returns the validation loss and the accuracy.
        """
        assert self.val_mask is not None, "Validation mask is not set! Please call :obj:`set_masks()` before executing training!"
        assert self.val_mask != 0, "Validation mask is set to 0! Please call :obj:`set_masks()` with different parameters!"


        self.model.eval()
        with torch.no_grad():
            out = self.model(val_data.x.to(self.device), val_data.edge_index.to(self.device))
            val_loss = self.criterion(out[val_mask], val_data.y[val_mask])
            _, pred = torch.max(out.data, 1)
            correct = (pred[val_mask] == val_data.y[val_mask]).float().sum().item()
            acc = correct / val_mask.shape[0]

    def train_validate(self,
                       train_data: Dataset | DataLoader,
                       val_data: Dataset | DataLoader,
                       epochs: int,
                       log: bool = False) -> tuple[list[float], list[float], list[float]]:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively as a tuple. If log is set to True, it will display
        the progress.
        """
        return super().train_validate(train_data=train_data,
                                      val_data=val_data,
                                      epochs=epochs,
                                      log=log)
    
    def set_masks(self,
                  data: InMemoryDataset | Dataset,
                  val_ratio: float = 0.2,
                  test_ratio: float = 1) -> None:
        """
        Creates the training, validation and test masks required for the pipeline to work.
        """
        assert val_ratio >= 0 and val_ratio < 1, "The ratio of the validation set to the whole dataset must be in [0,1)!"
        assert test_ratio > 0 and test_ratio <= 1, "The ratio of the test set to the whole dataset must be in (0,1]!"

        self.train_mask, self.val_mask, self.test_mask = \
            TrainValTestSplitter.single_graph_split(data, 
                                                    val_size=val_ratio, 
                                                    test_size=test_ratio)
