import torch
from torch_geometric.data import InMemoryDataset, Data

class SparseGraphDataset(InMemoryDataset):
    """An object representation of the synthetic datasets created by graphgen.GraphBuilder"""
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self._root: str = root
        self.num_classes = 5
        super(SparseGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> list[str]:
        return ['raw_dset_list.pt']
    
    @property
    def processed_file_names(self) -> list[str]:
        return ['processed_dset_list.pt']

    def download(self) -> None:
        print("""This function shouldn't be called, the datasets should 
            have been downloaded to their correct folder during the cloning process!e
            Repository link: https://git01lab.cs.univie.ac.at/peteri00/rewiring-heuristics.git""")

    def process(self):
        graphs: list[Data] = torch.load(f'{self._root}/{self.raw_file_names[0]}')
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx):
        return self._graphs[idx]