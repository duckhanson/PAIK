import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils.settings import config


def create_dataset(
    verbose: bool = False,
    features: np.array = None,
    targets: np.array = None,
    enable_normalize: bool = False,
    device: str = config.device,
):
    """
    _summary_

    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :param features: _description_, defaults to None
    :type features: np.array, optional
    :param targets: _description_, defaults to None
    :type targets: np.array, optional
    :param enable_normalize: _description_, defaults to False
    :type enable_normalize: bool, optional
    :param device: _description_, defaults to config.device
    :type device: str, optional
    :return: _description_
    :rtype: _type_
    """

    return CustomDataset(
        features, targets, device=device, enable_normalize=enable_normalize
    )


class CustomDataset(Dataset):
    def __init__(self, features, targets, device, enable_normalize):
        if len(features) != len(targets):
            raise ValueError("features and targets should have the same shape[0].")

        features = np.array(features)
        targets = np.array(targets)
        self.df = pd.DataFrame(data=np.column_stack((features, targets)))
        self.device = device
        self.features = torch.tensor(features, device=device, dtype=torch.float32)
        self.targets = torch.tensor(targets, device=device, dtype=torch.float32)
        if enable_normalize:
            self.normalize()

    def normalize(self):
        r_min = self.df.min(axis=0).to_numpy()
        r_max = self.df.max(axis=0).to_numpy()

        # use min max normalize
        self.features_min = torch.tensor(
            r_min[: self.features.shape[-1]], device=self.device, dtype=torch.float32
        )
        self.features_max = torch.tensor(
            r_max[: self.features.shape[-1]], device=self.device, dtype=torch.float32
        )
        self.targets_min = torch.tensor(
            r_min[self.features.shape[-1] :], device=self.device, dtype=torch.float32
        )
        self.targets_max = torch.tensor(
            r_max[self.features.shape[-1] :], device=self.device, dtype=torch.float32
        )

        self.features = (self.features - self.features_min) / (
            self.features_max - self.features_min
        )
        self.targets = (self.targets - self.targets_min) / (
            self.targets_max - self.targets_min
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, id):
        return self.features[id], self.targets[id]

    def create_loader(self, shuffle, batch_size):
        return DataLoader(self, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          drop_last=True,
                        #   generator=torch.Generator(device='cuda')
                          )
