import logging
import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import BaseTransform
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
class TrainValTestSplitter:
    r"""This class is used to get the training, validation and test masks for each dataset, that is used in the codebase."""

    def __init__(self) -> None:
        pass

    def mask_split(dataset_name: str, 
                   dataset: Dataset | InMemoryDataset, 
                   val_size: float = 0.2, 
                   test_size: float = 1) -> tuple[Tensor, Tensor, Tensor]:
        """
        Creates training, validation and test masks for a given dataset. Returns a tuple of these three tensors.
        """
        match dataset_name.lower():
            case 'cora': train_mask, val_mask, test_mask = TrainValTestSplitter._single_graph_split(dataset, val_size, test_size)
            case _: train_mask, val_mask, test_mask = TrainValTestSplitter._default_split_mask(dataset, val_size, test_size)
        return (train_mask, val_mask, test_mask)
    
    def dataset_split(dataset_name: str, 
                      dataset: Dataset | InMemoryDataset, 
                      val_size: float = 0.2, 
                      test_size: float = 1) -> tuple[Dataset, Dataset, Dataset]:
        """
        Creates training, validation and test dataset for a given dataset. Returns a tuple of the three datasets.
        """
        match dataset_name.lower():
            case 'pascal': train, val, test = TrainValTestSplitter._pascal_split()

        return (train, val, test)
    
    def _single_graph_split(dset: InMemoryDataset, 
                           val_size: float, 
                           test_size: float):
        """
        Splitting method that is used for single graph datasets where node classification is the task. (e.g. Planetoid)
        """
        log.debug(f"Proceeding with single graph splitting method")

        num_nodes = dset.y.shape[0]

        train_mask, val_mask = train_test_split(np.arange(num_nodes), test_size=val_size, shuffle=True)
        log.debug(f'Train mask: {train_mask}')
        log.debug(f'Val mask: {val_mask}')

        number_of_ones = int(num_nodes * test_size)
        number_of_zeros = num_nodes - number_of_ones

        permuted_indexes = torch.randperm(num_nodes)
        concated_tensor = torch.concat((torch.ones(number_of_ones, dtype=bool), torch.zeros(number_of_zeros, dtype=bool)))
        test_mask = concated_tensor[permuted_indexes]

        return (train_mask, val_mask, test_mask)
    
    def graph_class_split(
            data: Dataset | InMemoryDataset, 
            val_size: float = 0.2, 
            test_size: float = 1) -> tuple[Dataset | InMemoryDataset, Dataset | InMemoryDataset, Dataset | InMemoryDataset]:
        """
        Splitting method for graph classification tasks, it doesn't return masks, but the actual separate datasets.
        """
        train_size = int((1 - val_size) * len(data))
        val_size = len(data) - train_size
        test_size = int(test_size * len(data))
        not_test_size = len(data) - test_size

        train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
        if test_size == 1:
            test_data = data
        else:
            test_data, _ = torch.utils.data.random_split(data, [test_size, not_test_size])

        return (train_data, val_data, test_data)

    def _default_split_mask(data: Dataset | InMemoryDataset, val_size: float, test_size: float) -> tuple[Tensor, Tensor, Tensor]:
        log.debug(f'Proceeding with default splitting')
        if len(data) == 1:
            data = data[0]
        train_mask, test_mask = train_test_split(np.arange(len(data)), test_size=test_size)
        train_mask, val_mask = train_test_split(train_mask, test_size=val_size)

        train_mask = torch.tensor(train_mask)
        test_mask = torch.tensor(test_mask)
        val_mask = torch.tensor(val_mask)

        log.debug(f'Train mask: {train_mask}')
        log.debug(f'Val mask: {val_mask}')
        log.debug(f'Test mask: {test_mask}')

        return (train_mask, val_mask, test_mask)
    
    def pascal_split(transform: BaseTransform) -> tuple[InMemoryDataset, InMemoryDataset, InMemoryDataset]:
        train_dataset = LRGBDataset(root=f'Source/res/datasets/LRGB', 
                                    name='PascalVOC-SP', 
                                    split='train',
                                    pre_transform=transform)
        val_dataset = LRGBDataset(root=f'Source/res/datasets/LRGB', 
                                  name='PascalVOC-SP', 
                                  split='val', 
                                  pre_transform=transform)
        test_dataset = LRGBDataset(root=f'Source/res/datasets/LRGB', 
                                   name='PascalVOC-SP', 
                                   split='test', 
                                   pre_transform=transform)

        return(train_dataset, val_dataset, test_dataset)
