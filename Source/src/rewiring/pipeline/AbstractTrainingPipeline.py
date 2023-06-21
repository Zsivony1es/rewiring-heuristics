from abc import ABC, abstractmethod
from typing import Union

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

        self.model.to(self.device)
    
    @abstractmethod
    def single_train(self,
                     train_data: Union[Dataset, DataLoader]) -> float:
        """
        Trains the model on a given dataset for a single epoch. Returns the training loss.
        """

    @abstractmethod
    def single_validate(self,
                        val_data: Union[Dataset, DataLoader]) -> tuple:
        """
        Performs validation on the given dataset for a single epoch. Returns the validation loss and the accuracy.
        """

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
        training_loss = []
        validation_loss = []
        accuracy = []

        for epoch in range(1, epochs+1):
            # Training
            t_loss = self.single_train(train_data)
            training_loss.append(t_loss)
            # Validation
            v_loss, acc = self.single_validate(val_data)
            validation_loss.append(v_loss)
            accuracy.append(acc)
            if log:
                print(f"""Epoch: [{epoch}/{epochs}] \
        Training loss: {t_loss:.4f} \
        Validation loss: {v_loss:.4f} \
        Accuracy: {acc:.4f}""")

        return (training_loss, validation_loss, accuracy)