import logging

from torch_geometric import datasets

from graphgen.SparseGraphDataset import SparseGraphDataset


class Utils:
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    def get_rewiring_full_name(self, abbr: str) -> str:
        rew_names = ['AddRandomEdges', 'ReduceDiameter', 'GraphDiffusionConvolution', 'None']
        match abbr:
            case 'ARE': return rew_names[0]
            case 'RD': return rew_names[1]
            case 'GDC': return rew_names[2]
            case None: return rew_names[3]
            case _: raise ValueError(f'Invalid abbreviation: {abbr}')

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

        self._log.info(f"Beginning to load datasets\tRoot: {dset_base}")
        dsets = {}
        for dset in using_datasets:
            match dset.lower().strip():
                case 'cora': dsets['cora'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='Cora', transform=rewiring)
                case 'cora': dsets['citeseer'] = datasets.Planetoid(root=f'{dset_base}/Planetoid', name='CiteSeer', transform=rewiring)
                case 'enzymes': dsets['enzymes'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='ENZYMES', transform=rewiring)
                case 'proteins': dsets['proteins'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='PROTEINS', transform=rewiring)
                case 'mutag': dsets['mutag'] = datasets.TUDataset(root=f'{dset_base}/TUDataset', name='MUTAG', transform=rewiring)
                case 'synthetic': dsets['synthetic'] = SparseGraphDataset(root=f'{dset_base}/synthetic', transform=rewiring)
                case 'pascal': dsets['pascal'] = datasets.LRGBDataset(root=f'{dset_base}/LRGB', name='PascalVOC-SP', transform=rewiring)
                case _: raise ValueError(f'Invalid dataset setting: {dset}')
        print(f"Loading datasets DONE!")
        return dsets
