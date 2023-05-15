from src.rewiring.transform.ReduceDiameter import ReduceDiameter
import torch
from torch_geometric.data.data import Data
from torch_geometric import datasets
import networkx as nx
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("RDTransformTest")


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
    asd = torch.tensor(list(map(lambda tuple: tuple[1]['attr'], G.nodes.data())))

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
    asd = torch.tensor(list(map(lambda tuple: tuple[1]['attr'], G.nodes.data())))

    assert torch.all(torch.eq(asd, x)).item()
    assert str(G.__class__) == "<class 'networkx.classes.multidigraph.MultiDiGraph'>"

def test_rd_from_networkx_conversion_undirected():
    nodes = 6
    G = nx.MultiGraph()
    id = np.identity(nodes)
    for i in range(nodes):
        G.add_node(i, attr=id[i,:])
    edges_to_add = [(0, 1), (0, 4), (1, 3), (2, 3), (2, 4), (2, 5), (3, 5)]
    G.add_edges_from(edges_to_add)

    expected_x = torch.tensor([
        [0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 0., 0., 0., 0.],
        [2., 0., 0., 1., 0., 0., 0.],
        [3., 0., 0., 0., 1., 0., 0.],
        [4., 0., 0., 0., 0., 1., 0.],
        [5., 0., 0., 0., 0., 0., 1.]])
    
    expected_edge_index = torch.tensor([
        [0, 0, 1, 2, 2, 2, 3],
        [1, 4, 3, 3, 4, 5, 5]])
    
    rd = ReduceDiameter(max_diam=5)
    
    data = rd._from_networkx(G=G)

    assert torch.all(torch.eq(data.x, expected_x)).item()
    assert torch.all(torch.eq(data.edge_index, expected_edge_index)).item()
    assert str(G.__class__) == "<class 'networkx.classes.multigraph.MultiGraph'>"

def test_rd_conversion_integration_cora():
    dataset = datasets.Planetoid(root='../res/datasets/cora_tf', name='Cora')
    expected_data = dataset.get(0)

    rd = ReduceDiameter(5)
    G = rd._to_networkx(data=expected_data)
    data = rd._from_networkx(G=G)

    log.info(f"data.x parameters: {data.x}\n{expected_data.x}")

    assert torch.all(torch.eq(data.x, expected_data.x)).item()
    assert torch.all(torch.eq(data.edge_index, expected_data.edge_index)).item()
    assert torch.all(torch.eq(data.edge_attr, expected_data.edge_attr)).item()