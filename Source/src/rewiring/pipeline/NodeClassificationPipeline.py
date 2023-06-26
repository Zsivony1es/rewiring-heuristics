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
class NodeClassifiactionPipeline(AbstractTrainingPipeline):
    def __init__(self,
                device: torch.device,
                model: Module,
                criterion,
                optimizer) -> None:
        """
        Creates a pipeline designed to execute model training and evaluation for node classification tasks with multiple graphs in the dataset.
        """
        super().__init__(device=device,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer)

    def single_train(self,
                     train_data: Union[Dataset, DataLoader]) -> float:
        """
        Trains the model on a given dataset for one epoch. Returns the training loss.
        """
        assert isinstance(train_data, DataLoader)

        train_loss = []

        # Training loop
        self.model.train()
        self.model.to(self.device)
        total_loss = 0
        for ind, batch in enumerate(train_data):
            print(f"Batch [{ind}/{len(train_data)}]")
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch.x.to(self.device), batch.edge_index.to(self.device))
            loss = self.criterion(out.to(self.device), batch.y.to(self.device))
            loss.backward()
            train_loss.append(loss.item())

            self.optimizer.step()

        train_loss = np.array(train_loss).mean().item()

        return train_loss

    def single_validate(self,
                        val_data: Union[Dataset, DataLoader]) -> tuple:
        """
        Performs validation on the given dataset a single time. Returns the validation loss and the accuracy.
        """
        assert isinstance(val_data, DataLoader)

        val_loss = []
        correct = 0
        sum = len(val_data.dataset.y)

        self.model.eval()
        for batch in val_data:
            batch = batch.to(self.device)
            with torch.no_grad():
                out = self.model(batch.x.to(self.device), batch.edge_index.to(self.device))
                loss = self.criterion(out.to(self.device), batch.y.to(self.device))
                val_loss.append(loss.item())
                _, pred = torch.max(out.data.to(self.device), 1)
                correct += (pred.to(self.device) == batch.y.to(self.device)).float().sum().item()

        val_loss = np.array(val_loss).mean().item()
        acc = correct / sum

        return (val_loss, acc)

    def train_validate(self,
                       train_data: Union[Dataset, DataLoader],
                       val_data: Union[Dataset, DataLoader],
                       epochs: int,
                       log: bool = False) -> tuple:
        """
        Performs training and validation :obj:`epochs` number of times, returns the training,
        validation losses and the accuracy in each epoch respectively.
        """
        return super().train_validate(train_data=train_data,
                                      val_data=val_data,
                                      epochs=epochs,
                                      log=log)