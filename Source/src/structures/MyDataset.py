import torch
from torch_geometric.data import Dataset as Dataset

class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        print("Called download()...")

    def process(self):
        pass

    def len(self):
        pass

    def get(self, idx):
        pass