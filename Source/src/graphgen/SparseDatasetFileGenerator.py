import json
import logging

import numpy as np

from graphgen.SparseGraphBuilder import SparseGraphBuilder
from graphgen.SparseGraphDataset import SparseGraphDataset


log = logging.getLogger(__name__)
class SparseDatasetFileGenerator:
    """
    Used to generate the files of a pseudo-random dataset, that can be used by the :obj:`SparseGraphDataset` class.
    """
    
    def __init__(self, graph_count: int, json_path: str = None) -> None:
        assert graph_count > 0, "You must generate at least one graph!"
        assert graph_count < 10000, "Maximum number of graphs that can be generated is 10,000!"

        np.random.seed(15216142)

        if json_path is not None:
            self._conf = json.load(open(json_path,'r'))
        self._graph_count = graph_count
        
    def generate_dataset_files(self, 
                               out_path: str) -> None:
        r"""
        Used to generate the files required for a :obj:`SparseGraphDataset` object to be created.

        Parameters:
            :param:`out_path` (required) - Path to folder where the output files will be written
        """
        # Generated outside the for loop for ease of debugging
        node_counts = np.random.random_integers(60, 1000, self._graph_count)

        for i in range(self._graph_count):
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

            dset = graph_gen()

        
    