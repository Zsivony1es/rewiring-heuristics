import torch
from torch_geometric.data import Dataset

class GraphGenDataset(Dataset):
    """An object representation of the synthetic datasets created by Peter Ivony"""
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> list(str):
        return ['PI_dset1.node_attrs', 'PI_dset1.edges', 'PI_dset1.graph_lbls', 'PI_dset1.graph_idx',
                'PI_dset2.node_attrs', 'PI_dset2.edges', 'PI_dset2.graph_lbls', 'PI_dset2.graph_idx']
    
    @property
    def processed_file_names(self) -> list(str):
        return ['PI_dset1.pt', 'PI_dset2.pt']

    def download(self) -> None:
        raise NotImplementedError("""This function shouldn't be called, the datasets should 
            have been downloaded to their correct folder during the cloning process!\n
            Repository link: https://git01lab.cs.univie.ac.at/peteri00/rewiring-heuristics.git""")

    def process(self):
        pass

    def len(self):
        pass

    def get(self, idx):
        pass