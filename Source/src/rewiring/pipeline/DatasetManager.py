import logging

from torch_geometric import datasets
from torch_geometric.loader import DataLoader

from rewiring.TrainValTestSplitter import TrainValTestSplitter

from graphgen.SparseGraphDataset import SparseGraphDataset


log = logging.getLogger(__name__)
class DatasetManager:
    def __init__(self,
                 dataset_list: list[str],
                 transform = None) -> None:
        
        self.datasets: dict = self._get_dataset_dict(dataset_list, transform)
        train, val, test = self._get_loaders(dataset_list)

        self.train_loader = train
        self.val_loader = val
        self.test_loader = test

    def _get_dataset_dict(self,
                          dataset_list: list[str],
                          rewiring) -> dict:
        dset_base = "Source/res/datasets"

        log.info(f"Beginning to load datasets\tRoot: {dset_base}")
        dsets = {}
        for dset in dataset_list:
            match dset.lower().strip():
                case 'cora': dsets['cora'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='Cora', transform=rewiring)
                case 'citeseer': dsets['citeseer'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='CiteSeer', transform=rewiring)
                case 'enzymes': dsets['enzymes'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='ENZYMES', transform=rewiring, use_node_attr=True, use_edge_attr=True)
                case 'proteins': dsets['proteins'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='PROTEINS', transform=rewiring, use_node_attr=True, use_edge_attr=True)
                case 'mutag': dsets['mutag'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='MUTAG', transform=rewiring, use_node_attr=True, use_edge_attr=True)
                case 'synthetic': dsets['synthetic'] = SparseGraphDataset(root=f'{dset_base}/synthetic', transform=rewiring)
                case 'lrgb-pascal': 
                    dsets['lrgb-pascal'] = datasets.LRGBDataset(root=f'{dset_base}/LRGB', name='PascalVOC-SP', transform=rewiring)
                    dsets['lrgb-pascal_val'] = datasets.LRGBDataset(root=f'{dset_base}/LRGB', name='PascalVOC-SP', transform=rewiring, split='val')
                    dsets['lrgb-pascal_test'] = datasets.LRGBDataset(root=f'{dset_base}/LRGB', name='PascalVOC-SP', transform=rewiring, split='test')
                case _: raise ValueError(f'Invalid dataset setting: {dset}')
        print(f"Loading datasets DONE!")
        return dsets

    def _get_loaders(self,
                     dataset_list: list[str]) -> tuple[dict, dict, dict]:
        
        train_loader_dict = {}
        val_loader_dict = {}
        test_loader_dict = {}

        for dataset in dataset_list:
            
            if dataset in ['cora', 'citeseer']:
                # For cora and citeseer the training mask is used instead of a DataLoader object
                # So no entries are created for them
                continue
            elif dataset in ['proteins', 'mutag', 'enzymes']:
                train, val, test = self._get_tu_loaders(dataset)
            elif dataset == 'lrgb-pascal':
                train = DataLoader(self.datasets['lrgb-pascal'], batch_size=64, shuffle=True)
                val = DataLoader(self.datasets['lrgb-pascal_val'], batch_size=64, shuffle=False)
                test = DataLoader(self.datasets['lrgb-pascal_test'], batch_size=64, shuffle=False)
            elif dataset == 'synthetic':
                raise NotImplementedError("TODO: Get loaders for synthetic dataset")

            train_loader_dict[dataset] = train
            val_loader_dict[dataset] = val
            test_loader_dict[dataset] = test
            
        return (train_loader_dict, val_loader_dict, test_loader_dict)
    

    def _get_tu_loaders(self, 
                       dataset_name: str) -> tuple[dict, dict, dict]:
        
        train_dataset, val_dataset, test_dataset = TrainValTestSplitter.graph_class_split(
            self.datasets[dataset_name], 
            val_size=0.2, 
            test_size=1
        )

        train_dloader: DataLoader = DataLoader(train_dataset, 
                                               shuffle=True, 
                                               batch_size=64)

        val_dloader: DataLoader = DataLoader(val_dataset, 
                                            shuffle=False, 
                                            batch_size=64)
        
        test_dloader: DataLoader = DataLoader(test_dataset, 
                                            shuffle=False, 
                                            batch_size=64)
        
        return (train_dloader, val_dloader, test_dloader)