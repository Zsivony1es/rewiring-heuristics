import logging

import torch
import numpy as np
from torch_geometric.data import Data

from graphgen.SparseGraphBuilder import SparseGraphBuilder


log = logging.getLogger(__name__)
class SparseDatasetFileGenerator:
    """
    Used to generate the files of a pseudo-random dataset, that can be used by the :obj:`SparseGraphDataset` class.
    """
    
    def __init__(self, graph_count: int) -> None:
        assert graph_count > 0, "You must generate at least one graph!"
        assert graph_count < 10000, "Maximum number of graphs that can be generated is 10,000!"

        np.random.seed(15216142)
        self._graph_count = graph_count
        
    def generate_dataset_files(self, 
                               out_path: str) -> None:
        r"""
        Used to generate the files required for a :obj:`SparseGraphDataset` object to be created.

        Parameters:
            :param:`out_path` (required) - Path to folder where the output files will be written
        """

        dset_list = []

        for i in range(1, self._graph_count + 1):
            # Generate configuration
            cfg = {
                'directed': False,
                'multiple_edges': False,
                'loop_edges': True,
                'node_count_lim': 500,
                'node_feature_dim': 4,
                'density': 0.15,
                'edge_feature_dim': 1,
                'train_goal': 'node'
            }

            gen_seed = np.random.randint(300)
            graph_gen = SparseGraphBuilder(conf=cfg, seed=gen_seed)
            log.info(f"Creating graph [{i}/{self._graph_count}]")
            if i % 10 == 0:
                print(f"Creating graph [{i}/{self._graph_count}]")
            dset: Data = graph_gen()
            dset_list.append(dset)

        torch.save(dset_list, out_path)
            



        
    