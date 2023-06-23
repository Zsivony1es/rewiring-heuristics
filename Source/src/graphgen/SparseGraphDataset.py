import torch
from torch_geometric.data import InMemoryDataset, Data

class SparseGraphDataset(InMemoryDataset):
    """An object representation of the synthetic datasets created by graphgen.GraphBuilder"""
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SparseGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.root: str = root
        self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self) -> list:
        return ['raw_dset_list.pt']
    
    @property
    def processed_file_names(self) -> list:
        return ['pre_filter.pt', 'pre_transform.pt', 'processed_dset_list.pt']

    def download(self) -> None:
        print("""This function shouldn't be called, the datasets should 
            have been downloaded to their correct folder during the cloning process!e
            Repository link: https://git01lab.cs.univie.ac.at/peteri00/rewiring-heuristics.git""")

    def process(self):
        data_list: list = torch.load(f'{self.root}/{self.raw_file_names[0]}')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[2])
