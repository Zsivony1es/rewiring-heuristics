from src.rewiring.transform.ReduceDiameter import ReduceDiameter
import torch
from torch_geometric.data.data import Data


def test_rd_to_networkx_conversion_undirected():
    edge_index = torch.tensor([[0, 1, 1, 2, 1, 3, 2, 4],
                               [1, 0, 2, 1, 3, 1, 4, 2]], dtype=torch.float)
    x = torch.tensor([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    rd = ReduceDiameter(max_diam=5)

    G = rd._to_networkx(data=data)
    asd = torch.tensor(list((map(lambda tuple: tuple[1]['attr'], G.nodes.data()))))

    assert torch.all(torch.eq(asd, x)).item()
    assert str(G.__class__) == "<class 'networkx.classes.multigraph.MultiGraph'>"

def test_rd_to_networkx_conversion_directed():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 2, 3, 4]], dtype=torch.float)
    x = torch.tensor([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    rd = ReduceDiameter(max_diam=5)

    G = rd._to_networkx(data=data)
    asd = torch.tensor(list((map(lambda tuple: tuple[1]['attr'], G.nodes.data()))))

    assert torch.all(torch.eq(asd, x)).item()
    assert str(G.__class__) == "<class 'networkx.classes.multidigraph.MultiDiGraph'>"
