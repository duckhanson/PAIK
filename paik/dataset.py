import os
import numpy as np
import pandas as pd
import torch
from hnne import HNNE
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from paik.settings import config
from paik.utils import load_numpy, save_numpy


def _data_collection(robot, N: int, n: int, m: int, r: int):
    """
    collect data using uniform sampling

    :param robot: the robot arm you pick up
    :type robot: Robot
    :param N: #data required
    :type N: int
    :return: J, P
    :rtype: np.ndarray, np.ndarray
    """
    if N == config.N_train:
        path_J, path_P = config.path_J_train(n, m, r), config.path_P_train(
            n, m, r
        )  # type: ignore
    else:
        path_J, path_P = config.path_J_test(n, m, r), config.path_P_test(
            n, m, r
        )  # type: ignore

    J = load_numpy(file_path=path_J)
    P = load_numpy(file_path=path_P)

    if len(J) != N or len(P) != N:
        J, P = robot.sample_joint_angles_and_poses(n=N, return_torch=False)
        save_numpy(file_path=path_J, arr=J)
        save_numpy(file_path=path_P, arr=P[:, :m])

    return J, P


def load_all_data(robot, n, m, r):
    J_tr, P_tr = _data_collection(robot=robot, N=config.N_train, n=n, m=m, r=r)
    _, P_ts = _data_collection(robot=robot, N=config.N_test, n=n, m=m, r=r)
    F = _posture_feature_extraction(J=J_tr, P=P_tr, n=n, m=m, r=r)
    return J_tr, P_tr, P_ts, F


def _posture_feature_extraction(J: np.ndarray, P: np.ndarray, n: int, m: int, r: int):
    """
    generate posture feature from J (training data)

    Parameters
    ----------
    J : np.ndarray
        joint configurations
    P : np.ndarray
        poses of the robot

    Returns
    -------
    F : np.ndarray
        posture features
    """
    assert r > 0

    path = config.path_F(n, m, r)  # type: ignore
    F = load_numpy(file_path=path) if os.path.exists(path=path) else None

    if F is None or F.shape[-1] != r or len(F) != len(J):
        # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
        hnne = HNNE(dim=r)
        # maximum number of data for hnne (11M), we use max_num_data_hnne to test
        num_data = min(config.max_num_data_hnne, len(J))
        S = np.column_stack((J, P))
        F = hnne.fit_transform(X=S[:num_data], dim=r, verbose=True)
        # query nearest neighbors for the rest of J
        if len(F) != len(J):
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(S[:num_data])
            neigh_idx = knn.kneighbors(
                S[num_data:], n_neighbors=1, return_distance=False
            )
            neigh_idx = neigh_idx.flatten()  # type: ignore
            F = np.row_stack((F, F[neigh_idx]))

        save_numpy(file_path=path, arr=F)
    print(f"F load successfully from {path}")

    return F


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
