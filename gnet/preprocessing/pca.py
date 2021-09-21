from torch import nn
import torch

class PCA(nn.Module):

    def __init__(self, n_components:int=1):
        super().__init__()
        assert n_components<=3, 'Number of componenets exceed number of variables'
        self.n_components = n_components

    def forward(self, x):
        x = torch.transpose(x, -2, -1)  # x.shape = (batch, 4096, 3)
        *_, v = torch.pca_lowrank(x, q=3, center=False, niter=3)
        x = x @ v[..., :self.n_components]
        x = torch.transpose(x, -2, -1)   # x.shape = (batch, n_componenents, 4096 )
        return x
