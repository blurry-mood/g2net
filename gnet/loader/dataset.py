import os
import numpy as np
from torch.utils.data import Dataset

class RawDataset(Dataset):

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

class PredictDataset(Dataset):
    def __init__(self, paths) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ts = np.load(self.paths[i]).astype('float32')
        return os.path.split(self.paths[i])[-1].encode('utf-8'), ts # file name + signals