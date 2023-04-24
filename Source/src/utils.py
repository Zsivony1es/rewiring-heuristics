def get_rewiring_full_name(abbr: str) -> str:
    rew_names = ['AddRandomEdges', 'ReduceDiameter', 'DiffWire', 'None']
    match abbr:
        case 'ARE': return rew_names[0]
        case 'RD': return rew_names[1]
        case 'DW': return rew_names[2]
        case None: return rew_names[3]
        case _: raise ValueError(f'Invalid abbreviation: {abbr}')
