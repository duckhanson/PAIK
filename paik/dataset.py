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
    assert len(J) == len(P) and (F is None or len(P) == len(F))

    if F is None:
        C = P
    else:
        C = np.column_stack((P, F))

    dataset = create_dataset(features=J, targets=C, device=device)
    loader = dataset.create_loader(shuffle=True, batch_size=batch_size)

    return loader


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


def create_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    device: str,
    enable_normalize: bool = False,
):
    """
    _summary_

    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :param features: _description_, defaults to None
    :type features: np.ndarray, optional
    :param targets: _description_, defaults to None
    :type targets: np.ndarray, optional
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
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            #   generator=torch.Generator(device='cuda')
        )
