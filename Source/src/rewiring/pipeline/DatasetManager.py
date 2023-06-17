import logging

from torch_geometric import datasets
from torch.utils.data import DataLoader

from torch_geometric.data import InMemoryDataset, Dataset


from graphgen.SparseGraphDataset import SparseGraphDataset


log = logging.getLogger(__name__)
class DatasetManager:
    def __init__(self,
                 dataset_list: list[str],
                 transform = None) -> None:
        
        self.datasets: dict = DatasetManager._get_dataset_dict(dataset_list)
        train, val, test = DatasetManager._get_loaders(dataset_list)

        self.train_loader = train
        self.val_loader = val
        self.test_loader = test

    def _get_dataset_dict(dataset_list: list[str]) -> dict:
        pass

    def _get_loaders(dataset_list: list[str]) -> tuple[dict, dict, dict]:
        pass

    def get_dataset_and_loader(
            dataset_name: str) -> tuple[Dataset | InMemoryDataset, DataLoader, DataLoader, DataLoader]:
        
        if dataset_name == 'lrgb-pascal':
            train_dataset, train_loader, val_loader, test_loader = DatasetManager._get_pascal()
        else:
            pass

        return (train_dataset, train_loader, val_loader, test_loader)
    
    def _get_pascal(transform) -> tuple[Dataset | InMemoryDataset, DataLoader, DataLoader, DataLoader]:
        train_dataset = datasets.LRGBDataset(root=f'./../res/datasets/LRGB', name='PascalVOC-SP', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        val_dataset = datasets.LRGBDataset(root=f'./../res/datasets/LRGB', name='PascalVOC-SP', split='val')
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        test_dataset = datasets.LRGBDataset(root=f'./../res/datasets/LRGB', name='PascalVOC-SP', split='test')
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return (train_dataset, train_loader, val_loader, test_loader)
        
    def get_dataset_dictionary(self, dset_conf: str, rewiring) -> dict:
        dset_base = "Source/res/datasets"

        available_datasets = ['cora', 'citeseer', 'enzymes', 'proteins', 'mutag', 'lrgb-pascal', 'synthetic']
        using_datasets = None
        if dset_conf.strip().lower() == '*':
            using_datasets = available_datasets
        else:
            sep_list = dset_conf.strip().lower().split(',')
            for entered_dset in sep_list:
                if entered_dset not in available_datasets:
                    raise RuntimeError(f"Specified option for dataset: '{entered_dset}' does not exist!\nValid options: cora, enzymes, amazon, ppi")
            using_datasets = sep_list

        log.info(f"Beginning to load datasets\tRoot: {dset_base}")
        dsets = {}
        for dset in using_datasets:
            match dset.lower().strip():
                case 'cora': dsets['cora'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='Cora', transform=rewiring)
                case 'cora': dsets['citeseer'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='CiteSeer', transform=rewiring)
                case 'enzymes': dsets['enzymes'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='ENZYMES', transform=rewiring)
                case 'proteins': dsets['proteins'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='PROTEINS', transform=rewiring)
                case 'mutag': dsets['mutag'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='MUTAG', transform=rewiring)
                case 'synthetic': dsets['synthetic'] = SparseGraphDataset(root=f'{dset_base}/synthetic', transform=rewiring)
                case 'lrgb-pascal': dsets['lrgb-pascal'] = datasets.LRGBDataset(root=f'{dset_base}/LRGB', name='PascalVOC-SP', transform=rewiring)
                case _: raise ValueError(f'Invalid dataset setting: {dset}')
        print(f"Loading datasets DONE!")
        return dsets