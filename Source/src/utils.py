from torch_geometric import datasets
import logging

class Utils:
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)

    def get_rewiring_full_name(self, abbr: str) -> str:
        rew_names = ['AddRandomEdges', 'ReduceDiameter', 'DiffWire', 'None']
        match abbr:
            case 'ARE': return rew_names[0]
            case 'RD': return rew_names[1]
            case 'DW': return rew_names[2]
            case None: return rew_names[3]
            case _: raise ValueError(f'Invalid abbreviation: {abbr}')

    def get_dataset_dictionary(self, dset_conf: str, rewiring) -> dict:
        dset_base = "Source/res/datasets"

        available_datasets = ['cora', 'enzymes', 'amazon', 'ppi']
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
                case 'cora': dsets['cora'] = datasets.Planetoid(root=f'{dset_base}/cora_tf', name='Cora', transform=rewiring)
                case 'enzymes': dsets['enzymes'] = datasets.TUDataset(root=f'{dset_base}/enzymes', name='ENZYMES', transform=rewiring)
                case 'amazon': dsets['amazon'] = datasets.AmazonProducts(root=f'{dset_base}/amazonprods', transform=rewiring)
                case 'ppi': dsets['ppi'] = datasets.PPI(root=f'{dset_base}/ppi', transform=rewiring)
                case _: raise ValueError(f'Invalid dataset setting: {dset}')
        print(f"Loading datasets DONE!")
        return dsets
