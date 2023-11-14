import os
import pickle

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from utils.settings import config

def init_seeds(seed=42):
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        s = std_scale * esp * torch.rand((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, s))
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
    assert F is not None 
    P = np.atleast_2d(P[:, :m])
    F = np.atleast_2d(F)
    
    # Data Preprocessing: Posture Feature Extraction
    ref_F = np.atleast_2d(nearest_neighbor_F(knn, P, F, n_neighbors=k)) # type: ignore
    # ref_F = rand_F(P, F) # f_rand
    # ref_F = pick_F(P, F) # f_pick
    P = np.tile(P, (len(ref_F), 1)) if len(P) == 1 and k > 1 else P

    # Add noise std and Project to Tensor(device)
    C = torch.from_numpy(np.column_stack((P, ref_F, np.zeros((ref_F.shape[0], 1)))).astype(np.float32)).to(device=device) # type: ignore
    return C

def nearest_neighbor_F(knn: NearestNeighbors, P_ts: np.ndarray[float, float], F: np.ndarray[float, float], n_neighbors: int=1):
    if F is None:
        raise ValueError("F cannot be None")
    
    P_ts = np.atleast_2d(P_ts) # type: ignore
    assert len(P_ts) < len(F)
    # neigh_idx = knn.kneighbors(P_ts[:, :3], n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = knn.kneighbors(P_ts, n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = neigh_idx.flatten() # type: ignore
    
    return F[neigh_idx]

def rand_F(P_ts: np.ndarray, F: np.ndarray):
    return np.random.rand(len(np.atleast_2d(P_ts)), F.shape[-1])

def pick_F(P_ts: np.ndarray, F: np.ndarray) :
    idx = np.random.randint(low=0, high=len(F), size=len(np.atleast_2d(P_ts)))
    return F[idx]

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'model parameters: {pytorch_total_params}')


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