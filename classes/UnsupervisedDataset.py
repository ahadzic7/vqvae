from torch.utils.data import Dataset
import numpy as np
import subprocess
import csv
import os
  
class UnsupervisedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset, transform=None, labeling=False):
        self.base = base_dataset
        self.transform = transform
        self.labeling = labeling

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        datapoint = self.base[idx]
        if isinstance(datapoint, tuple):
            datapoint, label = datapoint
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint if not self.labeling else (datapoint,label)
