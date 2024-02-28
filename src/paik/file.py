import os
import shutil
import pickle
import numpy as np


# remove files under the directory
def remove_files_under_dir(dir_path, except_files=[]):
    """
    remove files under the directory
    exmaple of use: remove_files_under_dir('./wandb/', except_files=[])
    """

    for file in os.listdir(dir_path):
        if len(except_files) != 0 and file.split(".")[0] in except_files:
            continue

        file_path = os.path.join(dir_path, file)

        try:
            if os.path.isfile(file_path):
                print(f"remove {file_path}")
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                print(f"remove {file_path}")
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


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
        print(f"[ERROR] {file_path}: file not exist and return empty np array.")
        return None


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
