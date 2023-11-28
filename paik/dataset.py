import os
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def nearest_neighbor_F(
    knn: NearestNeighbors,
    P: np.ndarray,
    F: np.ndarray,
    n_neighbors: int = 1,
):
    assert F is not None
    return F[knn.kneighbors(np.atleast_2d(P), n_neighbors=n_neighbors, return_distance=False).flatten()]  # type: ignore


def rand_F(P: np.ndarray, F: np.ndarray):
    return np.random.rand(len(np.atleast_2d(P)), F.shape[-1])


def pick_F(P: np.ndarray, F: np.ndarray):
    return F[np.random.randint(low=0, high=len(F), size=len(np.atleast_2d(P)))]


class CustomDataset(Dataset):
    def __init__(self, features, targets):
        if len(features) != len(targets):
            raise ValueError("features and targets should have the same shape[0].")

        features = np.array(features)
        targets = np.array(targets)
        self.df = pd.DataFrame(data=np.column_stack((features, targets)))
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, id):
        return self.features[id], self.targets[id]
