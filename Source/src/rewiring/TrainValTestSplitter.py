import torch
from torch import Tensor
from torch_geometric.data import Dataset, InMemoryDataset
from sklearn.model_selection import train_test_split


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
        train_mask, val_mask = train_test_split(dset.y, test_size=0.1, stratify=dset.y)
        test_mask = torch.ones(dset.num_nodes, dtype=bool)
        return (train_mask, val_mask, test_mask)

    def _default_split(dset: Dataset | InMemoryDataset) -> tuple[Tensor, Tensor, Tensor]:
        train_mask, test_mask = train_test_split(range(dset.num_nodes), test_size=0.2)
        train_mask, val_mask = train_test_split(train_mask, test_size=0.2)

        train_mask = torch.tensor(train_mask).t()
        test_mask = torch.tensor(test_mask).t()
        val_mask = torch.tensor(val_mask).t()

        return (train_mask, val_mask, test_mask)
