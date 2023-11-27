import os
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


def get_train_loader(
    J: np.ndarray, P: np.ndarray, F: np.ndarray, batch_size: int, device: str
):
    """
    a training loader

    :param J: joint configurations
    :type J: np.ndarray
    :param P: end-effector positions
    :type P: np.ndarray
    :param F: posture features
    :type F: np.ndarray
    :return: torch dataloader
    :rtype: dataloader
    """
    assert len(J) == len(P) and F is not None

    return DataLoader(
        CustomDataset(features=J, targets=np.column_stack((P, F)), device=device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        #   generator=torch.Generator(device='cuda')
    )


def data_preprocess_for_inference(P, F, knn, m: int, k: int = 1, device: str = "cuda"):
    assert F is not None
    P = np.atleast_2d(P[:, :m])
    F = np.atleast_2d(F)

    # Data Preprocessing: Posture Feature Extraction
    ref_F = np.atleast_2d(nearest_neighbor_F(knn, P, F, n_neighbors=k))  # type: ignore
    # ref_F = rand_F(P, F) # f_rand
    # ref_F = pick_F(P, F) # f_pick
    P = np.tile(P, (len(ref_F), 1)) if len(P) == 1 and k > 1 else P

    # Add noise std and Project to Tensor(device)
    C = torch.from_numpy(
        np.column_stack((P, ref_F, np.zeros((ref_F.shape[0], 1)))).astype(np.float32)
    ).to(
        device=device
    )  # type: ignore
    return C


def nearest_neighbor_F(
    knn: NearestNeighbors,
    P_ts: np.ndarray[float, float],
    F: np.ndarray[float, float],
    n_neighbors: int = 1,
):
    if F is None:
        raise ValueError("F cannot be None")

    P_ts = np.atleast_2d(P_ts)  # type: ignore
    assert len(P_ts) < len(F)
    # neigh_idx = knn.kneighbors(P_ts[:, :3], n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = knn.kneighbors(P_ts, n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = neigh_idx.flatten()  # type: ignore

    return F[neigh_idx]


def rand_F(P_ts: np.ndarray, F: np.ndarray):
    return np.random.rand(len(np.atleast_2d(P_ts)), F.shape[-1])


def pick_F(P_ts: np.ndarray, F: np.ndarray):
    idx = np.random.randint(low=0, high=len(F), size=len(np.atleast_2d(P_ts)))
    return F[idx]


class CustomDataset(Dataset):
    def __init__(self, features, targets, device):
        if len(features) != len(targets):
            raise ValueError("features and targets should have the same shape[0].")

        features = np.array(features)
        targets = np.array(targets)
        self.df = pd.DataFrame(data=np.column_stack((features, targets)))
        self.device = device
        self.features = torch.tensor(features, device=device, dtype=torch.float32)
        self.targets = torch.tensor(targets, device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, id):
        return self.features[id], self.targets[id]
