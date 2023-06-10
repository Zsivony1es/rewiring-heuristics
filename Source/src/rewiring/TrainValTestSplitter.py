import logging
import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Dataset, InMemoryDataset
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
class TrainValTestSplitter:
    r"""This class is used to get the training, validation and test masks for each dataset, that is used in the codebase."""

    def __init__(self) -> None:
        pass

    def split(dataset_name: str, dataset: Dataset | InMemoryDataset) -> tuple[Tensor, Tensor, Tensor]:
        match dataset_name.lower():
            case 'cora': train_mask, val_mask, test_mask = TrainValTestSplitter._cora_split(dataset)
            case _: train_mask, val_mask, test_mask = TrainValTestSplitter._default_split(dataset)
        return (train_mask, val_mask, test_mask)
    
    def _cora_split(dset: InMemoryDataset):
        log.debug(f"Proceeding with 'cora' splitting method")
        train_mask, val_mask = train_test_split(np.arange(dset.y.shape[0]), test_size=0.2, shuffle=True)
        log.debug(f'Train mask: {train_mask}')
        log.debug(f'Val mask: {val_mask}')
        test_mask = torch.ones(dset.num_nodes, dtype=bool)
        return (train_mask, val_mask, test_mask)

    def _default_split(dset: Dataset | InMemoryDataset) -> tuple[Tensor, Tensor, Tensor]:
        log.debug(f'Proceeding with default splitting')
        if len(dset) == 1:
            dset = dset[0]
        train_mask, test_mask = train_test_split(np.arange(len(dset)), test_size=0.2)
        train_mask, val_mask = train_test_split(train_mask, test_size=0.2)

        train_mask = torch.tensor(train_mask)
        test_mask = torch.tensor(test_mask)
        val_mask = torch.tensor(val_mask)

        log.debug(f'Train mask: {train_mask}')
        log.debug(f'Val mask: {val_mask}')
        log.debug(f'Test mask: {test_mask}')

        return (train_mask, val_mask, test_mask)
