import logging

log = logging.getLogger(__name__)
class Utils:
    def get_rewiring_full_name(abbr: str) -> str:
        log.debug(f'Getting full name for {abbr}')
        rew_names = ['AddRandomEdges', 'ReduceDiameter', 'GraphDiffusionConvolution', 'None']
        match abbr:
            case 'ARE': return rew_names[0]
            case 'RD': return rew_names[1]
            case 'GDC': return rew_names[2]
            case None: return rew_names[3]
            case _: raise ValueError(f'Invalid abbreviation: {abbr}')

    def get_dataset_list(input: str) -> list[str]:

        available_datasets = ['cora', 'citeseer', 'enzymes', 'proteins', 'mutag', 'lrgb-pascal', 'synthetic']

        if input == "*":
            return available_datasets
        else:
            dset_list = list(map(lambda str: str.strip(), input.split(',')))

        for entered_dset in dset_list:
            if entered_dset not in available_datasets:
                raise RuntimeError(f"Specified option for dataset: '{entered_dset}' does not exist!\nValid options: cora, enzymes, amazon, ppi")

        return dset_list