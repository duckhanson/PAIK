import os
import pickle
import numpy as np
import torch
from utils.settings import config


def init_seeds(seed=42):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed
    )  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed
    )  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

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


def normalize(arr: np.ndarray, arr_min: np.ndarray, arr_max: np.ndarray):
    # norm = (val - min) / (max - min)
    return (arr - arr_min) / (arr_max - arr_min)


def denormalize(norm: np.ndarray, arr_min: np.ndarray, arr_max: np.ndarray):
    # example of use (the only one neede denormalize):
    # joint config = model_output * (joint_max - joint_min) + joint_min
    # arr = norm * (arr_max - arr_min) + arr_min
    return norm * (arr_max - arr_min) + arr_min


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {pytorch_total_params}")


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
