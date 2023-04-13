import torch
import torch.nn.functional as F

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class FeatureDegree(BaseTransform):
    r"""Code taken from: https://github.com/AdrianArnaiz/DiffWire/blob/5aded812451639187680c2d70bf16f4d2f21ca2e/transforms/transform_features.py
    """
    def __init__(self, in_degree=False, cat=True):
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.float).unsqueeze(-1)
        #deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_degree})'