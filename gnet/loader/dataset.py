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