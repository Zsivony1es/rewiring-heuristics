from abc import ABC, abstractmethod

import torch
from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class AbstractTrainingPipeline(ABC):
    def __init__(self,
                device: torch.device,
                model: Module,
                criterion,
                optimizer) -> None:
        """
        An abstract class, whose child classes implement a training pipeline for some 
        specific learning task (e.g. graph classification, node classification, etc.)
        """
        super().__init__()
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    
    @abstractmethod
    def single_train(self,
              train_dset: Dataset | DataLoader) -> float:
        """
        Trains the model on a given dataset for a single epoch. Returns the training loss.
        """

    @abstractmethod
    def single_validate(self,
                 val_dataset: Dataset) -> tuple[float, float]:
        """
        Performs validation on the given dataset for a single epoch. Returns the validation loss and the accuracy.
        """
    
    def train_validate(self,
                       train_dloader: DataLoader,
                       val_dloader: DataLoader,
                       epochs: int,
                       log: bool = False) -> tuple[list[float], list[float], list[float]]:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively as a tuple. If log is set to True, it will display
        the progress.
        """
        training_loss = []
        validation_loss = []
        accuracy = []

        for epoch in epochs:
            # Training
            t_loss = self.single_train(train_dloader)
            training_loss.append(t_loss)
            # Validation
            v_loss, acc = self.single_validate(val_dloader)
            validation_loss.append(v_loss)
            accuracy.append(acc)

        return (training_loss, validation_loss, accuracy)