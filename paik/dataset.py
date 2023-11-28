import os
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset




def data_preprocess_for_inference(P, F, knn, m: int, k: int = 1):
    assert F is not None
    P = np.atleast_2d(P[:, :m])
    F = np.atleast_2d(F)

    # Data Preprocessing: Posture Feature Extraction
    ref_F = np.atleast_2d(nearest_neighbor_F(knn, P, F, n_neighbors=k))  # type: ignore
    # ref_F = rand_F(P, F) # f_rand
    # ref_F = pick_F(P, F) # f_pick
    P = np.tile(P, (len(ref_F), 1)) if len(P) == 1 and k > 1 else P

    # Add noise std
    return torch.from_numpy(
        np.column_stack((P, ref_F, np.zeros((ref_F.shape[0], 1)))).astype(np.float32)
    )


def nearest_neighbor_F(
    knn: NearestNeighbors,
    P: np.ndarray[float, float],
    F: np.ndarray[float, float],
    n_neighbors: int = 1,
):
    if F is None:
        raise ValueError("F cannot be None")

    P = np.atleast_2d(P)  # type: ignore
    assert len(P) < len(F)
    return F[knn.kneighbors(P, n_neighbors=n_neighbors, return_distance=False).flatten()]  # type: ignore


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
