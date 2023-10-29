import os
# from tqdm.auto import tqdm
from tqdm import trange, tqdm
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from hnne import HNNE
from sklearn.neighbors import NearestNeighbors
from jrl.evaluation import solution_pose_errors
from utils.dataset import create_dataset
from utils.settings import config

def init_seeds(seed=42):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def data_collection(robot, N: int, n: int, m: int, r: int):
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
        path_J, path_P = config.path_J_train(n, m, r), config.path_P_train(n, m, r)
    else:
        path_J, path_P = config.path_J_test(n, m, r), config.path_P_test(n, m, r)

    J = load_numpy(file_path=path_J)
    P = load_numpy(file_path=path_P)

    if len(J) != N or len(P) != N:
        J, P = robot.sample_joint_angles_and_poses(n=N, return_torch=False)
        save_numpy(file_path=path_J, arr=J)
        save_numpy(file_path=path_P, arr=P[:, :m])

    return J, P

def load_all_data(robot, n, m, r):
    J_tr, P_tr = data_collection(robot=robot, N=config.N_train, n=n, m=m, r=r)
    _, P_ts = data_collection(robot=robot, N=config.N_test, n=n, m=m, r=r)
    F = posture_feature_extraction(J=J_tr, P=P_tr, n=n, m=m, r=r)
    return J_tr, P_tr, P_ts, F


def posture_feature_extraction(J: np.ndarray, P: np.ndarray, n: int, m: int, r: int):
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
    F = None
    
    if r == 0:
        return F
    
    path = config.path_F(n, m, r)
    if os.path.exists(path=path):
        F = load_numpy(file_path=path)
    
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
            neigh_idx = knn.kneighbors(S[num_data:], n_neighbors=1, return_distance=False)
            neigh_idx = neigh_idx.flatten() # type: ignore
            F = np.row_stack((F, F[neigh_idx]))

        save_numpy(file_path=path, arr=F)
    print(f"F load successfully from {path}")

    return F


def get_train_loader(J: np.ndarray, P: np.ndarray, F: np.ndarray, batch_size: int, device: str):
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

def load_pickle(file_path: str):
    """
    load pickle file, e.g. hnne, knn model

    Parameters
    ----------
    file_path : str
        file_path = file_name.pickle

    Returns
    -------
    object
        loaded model

    Raises
    ------
    FileNotFoundError
        file_path not exists
    """
    if os.path.exists(path=file_path):
        with open(file_path, "rb") as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
        return data
    else:
        raise FileNotFoundError(f"{file_path}: file not exist and return None.")

def save_pickle(file_path: str, obj):
    """
    save object as a pickle file

    Parameters
    ----------
    file_path : str
        file_path = file_name.pickle
    obj : Object
        anytype of models
    """
    dir_path = os.path.dirname(p=file_path)
    if not os.path.exists(path=dir_path):
        print(f"mkdir {dir_path}")
        os.mkdir(path=dir_path)

    with open(file_path, "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_numpy(file_path: str):
    """
    load numpy file

    Parameters
    ----------
    file_path : str
        file_path = file_name.np

    Returns
    -------
    np.array
        loaded data
    """
    if os.path.exists(path=file_path):
        return np.load(file_path)
    else:
        print(f"{file_path}: file not exist and return empty np array.")
        return np.array([])

def save_numpy(file_path: str, arr: np.ndarray):
    """
    save arr as a numpy file

    Parameters
    ----------
    file_path : str
        file_path = file_name.np
    arr : np.ndarray
        data
    """

    dir_path = os.path.dirname(p=file_path)
    if not os.path.exists(path=dir_path):
        print(f"mkdir {dir_path}")
        os.mkdir(path=dir_path)
    np.save(file_path, arr)

def add_noise(batch, esp: float, std_scale: float):
    J, C = batch
    if esp < 1e-9:
        std = torch.zeros((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, std))
    else:
        # softflow implementation
        s = esp * torch.rand((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, s * std_scale))
        noise = torch.normal(
            mean=torch.zeros_like(input=J),
            std=torch.repeat_interleave(input=s, repeats=J.shape[1], dim=1),
        )
        J = J + noise
    return J, C

def normalize(arr: np.ndarray, arr_min: np.ndarray, arr_max: np.ndarray):
    # norm = (val - min) / (max - min)
    return (arr - arr_min) / (arr_max - arr_min)

def denormalize(norm: np.ndarray, arr_min: np.ndarray, arr_max: np.ndarray):
    # example of use (the only one neede denormalize):
    # joint config = model_output * (joint_max - joint_min) + joint_min
    # arr = norm * (arr_max - arr_min) + arr_min
    return norm * (arr_max - arr_min) + arr_min

def data_preprocess_for_inference(P, F, knn, m: int, k: int=1, device: str = 'cuda'):
    P = np.atleast_2d(P)
    F = np.atleast_2d(F)
    
    if m == 3:
        P = P[:, :3]
    if F is not None:
        # Data Preprocessing: Posture Feature Extraction
        ref_F = nearest_neighbor_F(knn, P, F, n_neighbors=k) # knn
        ref_F = np.atleast_2d(ref_F) # type: ignore
        # ref_F = rand_F(P, F) # f_rand
        # ref_F = pick_F(P, F) # f_pick
        if len(P) == 1 and k > 1:
            P = np.tile(P, (len(ref_F), 1))
        C = np.column_stack((P, ref_F))
    else:
        C = P
        
    C = C.astype(np.float32)

    # Project to Tensor(device)
    C = torch.from_numpy(C).to(device)
    _, C = add_noise((torch.zeros_like(C), C), esp=0)

    return C

def nearest_neighbor_F(knn: NearestNeighbors, P_ts: np.ndarray[float, float], F: np.ndarray[float, float], n_neighbors: int=1):
    if F is None:
        raise ValueError("F cannot be None")
    
    P_ts = np.atleast_2d(P_ts) # type: ignore
    assert len(P_ts) < len(F)
    neigh_idx = knn.kneighbors(P_ts[:, :3], n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = neigh_idx.flatten() # type: ignore
    
    return F[neigh_idx]

def rand_F(P_ts: np.ndarray, F: np.ndarray):
    return np.random.rand(len(np.atleast_2d(P_ts)), F.shape[-1])

def pick_F(P_ts: np.ndarray, F: np.ndarray) :
    idx = np.random.randint(low=0, high=len(F), size=len(np.atleast_2d(P_ts)))
    return F[idx]


def evaluate_solver(robot, solver, P_ts, F, knn, K=10):
    C = data_preprocess_for_inference(P=P_ts, F=F, knn=knn, m=solver._m)

    with torch.inference_mode():
        J_hat = solver(C).sample((K,))

    l2_errs = np.empty((J_hat.shape[0], J_hat.shape[1]))
    ang_errs = np.empty((J_hat.shape[0], J_hat.shape[1]))
    if P_ts.shape[-1] == 3:
        P_ts = np.column_stack((P_ts, np.ones(shape=(len(P_ts), 4))))
    for i, J in enumerate(J_hat):
        l2_errs[i], ang_errs[i] = solution_pose_errors(robot=robot, solutions=J, target_poses=P_ts)
        
    l2_errs = l2_errs.flatten()
    ang_errs = ang_errs.flatten()

    # df = pd.DataFrame()
    # df['l2_errs'] = l2_errs
    # df['ang_errs'] = ang_errs
    # print(df.describe())
    return l2_errs.mean(), ang_errs.mean()

def create_robot_dirs() -> None:
    """
    _summary_
    """
    for dp in config.dir_paths:
        if not os.path.exists(path=dp):
            os.makedirs(name=dp)
            print(f"Create {dp}")
            
def remove_empty_robot_dirs() -> None:
    """
    _summary_
    """
    for dp in config.dir_paths:
        walk = list(os.walk(dp))
        for p, _, _ in walk[::-1]:
            if os.path.exists(p) and len(os.listdir(p)) == 0:
                os.removedirs(p)
                print(f"Delete {p}")

class EarlyStopping:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, enable_save=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.enable_save = enable_save
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {score}, best: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.enable_save:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
        else:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).')
        self.val_loss_min = val_loss