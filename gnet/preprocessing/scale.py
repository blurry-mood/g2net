from torch import nn

class Scale(nn.Module):
    def forward(self, x):
        x = (x - x.mean(2, keepdim=True))/x.std(2, keepdim=True)
        return x