from graphgen.GraphBuilder import GraphBuilder
from graphgen.GraphGenDataset import GraphGenDataset
from torch_geometric.data import Dataset
import json


class DatasetFactory:
    """Used to generate a pseudo-random :obj:`MyDataset`."""
    
    def __init__(self, json_path: str, graph_count: int) -> None:
        assert json_path is not None, "The path to the JSON file must be set!"
        assert graph_count > 0, "You must generate at least one graph!"
        assert graph_count < 10000, "Maximum number of graphs that can be generated is 10,000!"

        self._randomizer_seed = 15216142
        self._conf = json.load(open(json_path,'r'))
        self._graph_count = graph_count
        self._graph_gen = GraphBuilder(
            conf=self._conf,
            rand_seed=self._randomizer_seed)
        
    def generate_dataset(self) -> GraphGenDataset:
        for _ in range(self._graph_count):
            self._graph_gen.build()
        pass
        
    