from torch import nn
import timm
import torch

def model(model_name, pretrained, num_classes):
    return timm.create_model(model_name, pretrained, num_classes=num_classes,  )


class Paper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 1),
            nn.Tanh()
        )

        self.decoder = nn.LSTM(input_size=64, hidden_size=100, num_layers=3, batch_first=True, bidirectional=True, dropout=0.)

        self.dense = nn.Linear(100, 1)
        self.dense2 = nn.Linear(512, num_classes)

    def forward(self, x:torch.Tensor):
        b = x.size(0)
        # x.shape = (batch, 3, 4096)
        x = x.unflatten(-1, (8, 512)).permute(0, 3, 1, 2).flatten(0, 1)
        # x.shape = (batch * 512, 3, 8)
        
        x = self.encoder(x)   # shape=(batch * 512, 16, 4)
        x = x.flatten(-2, -1).unflatten(0, (b, 512))   # shape = (batch, 512, 64)
        x, *_ = self.decoder(x) # x.shape = (batch, 512, 2*100)
        x = x.view(-1, 512, 2, 100)[:,:,0]  # x.shape = (batch, 512, 100)
        x = self.dense(x).squeeze(-1)   # x.shape = (batch, 512)
        x = self.dense2(x)  # x.shape = (batch, num_classes)
        return x