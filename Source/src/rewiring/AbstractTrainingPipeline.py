from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class AbstractTrainingPipeline(ABC):
    @abstractmethod
    def __init__(self,
                device: torch.device,
                model: Module,
                criterion,
                optimizer) -> None:
        super().__init__()
    
    @abstractmethod
    def single_train(self,
              train_dset: Dataset | DataLoader) -> float:
        """
        Trains the model on a given dataset for one epoch. Returns the training loss.
        """

    @abstractmethod
    def single_validate(self,
                 val_dataset: Dataset) -> tuple[float, float]:
        """
        Performs validation on the given dataset a single time. Returns the validation loss and the accuracy.
        """

    @abstractmethod
    def train_validate(self,
                       epochs: int) -> tuple[list[float], list[float], list[float]]:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively.
        """