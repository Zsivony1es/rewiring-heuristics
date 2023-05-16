from src.rewiring.transform.ReduceDiameter import ReduceDiameter
import torch
from torch_geometric.data.data import Data
from torch_geometric import datasets
import networkx as nx
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG)
# log = logging.getLogger("RDTransformTest")


def _sort_edge_index(edge_index):
    sorted_indices = torch.argsort(edge_index[0])
    sorted_tensor = edge_index[:, sorted_indices]

    repeated_indices = torch.where(sorted_tensor[0][1:] == sorted_tensor[0][:-1])[0]

    # Check if there are repeated values
    if repeated_indices.numel() > 0:
        # Get start and end indices for each group of repeated values
        start_indices = torch.cat((torch.tensor([0]), repeated_indices + 1))
        end_indices = torch.cat((repeated_indices + 1, torch.tensor([sorted_tensor.shape[1]])))

        # Sort the corresponding values in the second array within each group
        for start, end in zip(start_indices, end_indices):
            group_sorted_indices_second = torch.argsort(sorted_tensor[1, start:end])
            sorted_tensor[:, start:end] = sorted_tensor[:, start:end][:, group_sorted_indices_second]

    return sorted_tensor


def test_rd_is_multigraph():
    edge_index1 = torch.tensor([
        [0, 1, 0, 1, 2],
        [1, 0, 1, 0, 1]])
    edge_index2 = torch.tensor([
        [0, 1, 2],
        [1, 0, 1]])
    
    data = Data(x=None, edge_index=edge_index1)
    
    assert ReduceDiameter._is_multigraph(edge_index=edge_index1)
    assert ReduceDiameter._is_multigraph(data=data)
    assert not ReduceDiameter._is_multigraph(edge_index=edge_index2)



def test_rd_to_networkx_conversion_undirected():
    edge_index = torch.tensor([[0, 1, 1, 2, 1, 3, 2, 4],
                               [1, 0, 2, 1, 3, 1, 4, 2]], dtype=torch.float)
    x = torch.tensor([[0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [2, 0, 0, 1, 0, 0],
                      [3, 0, 0, 0, 1, 0],
                      [4, 0, 0, 0, 0, 1]], dtype=torch.float)
    
    y = torch.tensor([1, 2, 3, 4, 5])
    rd = ReduceDiameter(5)
    data = Data(x=x, edge_index=edge_index, y=y)

    G = rd._to_networkx(data=data)
    asd = torch.tensor(list(map(lambda tuple: tuple[1]['feat'], G.nodes.data())))

    assert torch.all(torch.eq(asd, x)).item()
    assert str(G.__class__) == "<class 'networkx.classes.graph.Graph'>"

def test_rd_to_networkx_conversion_directed():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 2, 3, 4]], dtype=torch.float)
    x = torch.tensor([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)

    G = ReduceDiameter._to_networkx(data=data)
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
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 1.]])
    
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
    
    G = ReduceDiameter._to_networkx(data=expected_data)

    data = ReduceDiameter._from_networkx(G=G)

    expected_edge_index = _sort_edge_index(expected_data.edge_index)
    got_edge_ind = _sort_edge_index(data.edge_index)

    print(f"data.edge_index parameters:\nReceived:\n{got_edge_ind.tolist()}\nExpected:\n{expected_edge_index.tolist()}")

    assert torch.all(torch.eq(data.x, expected_data.x)).item()
    assert got_edge_ind.shape == expected_edge_index.shape
    assert torch.all(torch.eq(got_edge_ind, expected_edge_index)).item()
    assert torch.all(torch.eq(data.edge_attr, expected_data.edge_attr)).item()