from torch import nn
import timm
import torch

def model(model_name, pretrained, num_classes):
    return timm.create_model(model_name, pretrained, num_classes=num_classes,  )


class Paper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

        self.decoder = nn.LSTM(input_size=1024, hidden_size=100, num_layers=3, batch_first=True, bidirectional=True, dropout=0.)

        self.dense = nn.Linear(100, 1)
        self.dense2 = nn.Linear(256, num_classes)

    def forward(self, x:torch.Tensor):
        b = x.size(0)
        # x.shape = (batch, 3, 256, 257)
        x = x.permute(0, 2, 1, 3).flatten(0, 1)
        # x.shape = (batch * 256, 3, 257)
        
        x = self.encoder(x)   # shape=(batch * 256, 16, 64)
        x = x.flatten(-2, -1).unflatten(0, (b, 256))   # shape = (batch, 256, 1024)
        x, *_ = self.decoder(x) # x.shape = (batch, 256, 2*100)
        x = x.view(-1, 256, 2, 100)[:,:,0]  # x.shape = (batch, 256, 100)
        x = self.dense(x).squeeze(-1)   # x.shape = (batch, 256)
        x = self.dense2(x)  # x.shape = (batch, num_classes)
        return x