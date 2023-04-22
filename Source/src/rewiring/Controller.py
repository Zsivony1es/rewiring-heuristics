import torch
import logging
import argparse

import rewiring.model
from rewiring.transform.AddRandomEdges import AddRandomEdges
from rewiring.transform.ReduceDiameter import ReduceDiameter
from rewiring.transform.DiffWire import DiffWire
from rewiring.PlotGenerator import PlotGenerator
from torch_geometric import datasets
from utils import get_rewiring_full_name

class Controller():

    def __init__(self) -> None:
        self._plot_generator = PlotGenerator()
        self._rewiring_strategy = None
        self._loss_function = torch.nn.NLLLoss()
        pass



    def get_rewiring_strategy(self) -> None:
        pass