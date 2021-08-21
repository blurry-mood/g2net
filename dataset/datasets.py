import numpy as np
from torch.utils.data import Dataset
import torch

class TransformDataset(Dataset):

    def __init__(self, paths, transform):
        super().__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ts = np.load(self.paths[i]).astype('float32')
        ts = torch.from_numpy(ts)
        specs = self.transform(ts)
        return i, specs

class SpecDataset(Dataset):

    def __init__(self, paths, labels):
        super().__init__()
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ts = np.load(self.paths[i]).astype('float32')
        label = self.labels[i]
        return ts, label