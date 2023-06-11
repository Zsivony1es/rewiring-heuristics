import torch

from rewiring.AbstractTrainingPipeline import AbstractTrainingPipeline


class GraphClassificationPipeline(AbstractTrainingPipeline):
    def __init__(self, 
                 device: torch.device) -> None:
        pass