import torch
import torch_geometric

from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from rewiring.pipeline.AbstractTrainingPipeline import AbstractTrainingPipeline


class NodeClassifiactionPipeline(AbstractTrainingPipeline):
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
              train_dset: Dataset | DataLoader) -> float:
        """
        Trains the model on a given dataset for one epoch. Returns the training loss.
        """
        pass

    def single_validate(self,
                 val_dataset: Dataset | DataLoader) -> tuple[float, float]:
        """
        Performs validation on the given dataset a single time. Returns the validation loss and the accuracy.
        """
        pass

    def train_validate(self,
                       train_data: Dataset | DataLoader,
                       val_data: Dataset | DataLoader,
                       epochs: int,
                       log: bool = False) -> tuple[list[float], list[float], list[float]]:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively.
        """
        return super().train_validate(train_data=train_data,
                                      val_data=val_data,
                                      epochs=epochs,
                                      log=log)