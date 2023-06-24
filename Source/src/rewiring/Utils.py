import logging
import shutil


log = logging.getLogger(__name__)
class Utils:
    def get_rewiring_full_name(abbr: str) -> str:
        log.debug(f'Getting full name for {abbr}')
        rew_names = ['AddRandomEdges', 'ReduceDiameter', 'GraphDiffusionConvolution', 'None']
        if abbr == 'ARE':
            return rew_names[0]
        elif abbr == 'RD': 
            return rew_names[1]
        elif abbr == 'GDC': 
            return rew_names[2]
        elif abbr is None: 
            return rew_names[3]
        else:
            raise ValueError(f'Invalid abbreviation: {abbr}')

    def get_dataset_list(input: str) -> list:

        available_datasets = ['cora', 'citeseer', 'enzymes', 'proteins', 'mutag', 'lrgb-pascal', 'synthetic']

        if input == "*":
            return available_datasets
        else:
            dset_list = list(map(lambda str: str.strip(), input.split(',')))

        for entered_dset in dset_list:
            if entered_dset not in available_datasets:
                raise RuntimeError(f"Specified option for dataset: '{entered_dset}' does not exist!\nValid options: cora, enzymes, amazon, ppi")

        return dset_list
    
    def delete_processed_folder(dataset_name: str) -> None:
        """
        Deletes the processed folder of the entered dataset.
        """
        base_path = "Source/res/datasets"
        if dataset_name.lower() in ['cora', 'citeseer']:
            path = base_path + "/Planetoid/"
            if dataset_name.lower() == 'cora':
                path = path + "Cora"
            else:
                path = path + "CiteSeer"
        elif dataset_name.lower() in ['proteins', 'mutag', 'enzymes']:
            path = base_path + f"/TUDataset/{dataset_name.upper()}"
        elif dataset_name.lower() == 'lrgb-pascal':
            path = base_path + "/LRGB/pascalvoc-sp"
        elif dataset_name.lower() == 'synthetic':
            path = base_path + "/synthetic"

        path = path + "/processed"

        shutil.rmtree(path)
