import os
# from tqdm.auto import tqdm
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance # distance.cosine as cosine_distance
from hnne import HNNE
from sklearn.neighbors import NearestNeighbors

from utils.dataset import create_dataset
from utils.settings import config


def _get_data_path(len_data):
    """
    _summary_

    :param len_data: _description_
    :type len_data: _type_
    :return: (x_path, y_path, x_trans_path)
    :rtype: _type_
    """
    if len_data == config.N_train:
        return config.path_J_train, config.path_P_train
    else:
        return config.path_J_test, config.path_P_test


def data_collection(robot, N: int):
    """
    collect data using uniform sampling

    :param robot: the robot arm you pick up
    :type robot: Robot
    :param N: #data required
    :type N: int
    :return: J, P
    :rtype: np.array, np.array
    """
    assert config.m == 3 or config.m == 3 + 4
    
    path_J, path_P = _get_data_path(len_data=N)

    J = load_numpy(file_path=path_J)
    P = load_numpy(file_path=path_P)

    if len(J) != N or len(P) != N:
        if config.m == 3:
            J, P = robot.uniform_sample_J(num_samples=N)
        else:
            J, P = robot.uniform_sample_J_quaternion(num_samples=N)
        save_numpy(file_path=path_J, arr=J)
        save_numpy(file_path=path_P, arr=P)

    return J, P

def load_all_data(robot):
    J_tr, P_tr = data_collection(robot=robot, N=config.N_train)
    _, P_ts = data_collection(robot=robot, N=config.N_test)
    F = posture_feature_extraction(J=J_tr)
    
    if config.enable_normalize:
        J_tr = normalize(J_tr, robot.joint_min, robot.joint_max)
        P_ts = normalize(P_ts, P_tr.min(axis=0), P_tr.max(axis=0))
        P_tr = normalize(P_tr, P_tr.min(axis=0), P_tr.max(axis=0))
        F = normalize(F, F.min(axis=0), F.max(axis=0))
        print("Load normalize data. [NOTICE] Please check out denormalize works !")
    
    return J_tr, P_tr, P_ts, F
        


def posture_feature_extraction(J: np.array):
    """
    generate posture feature from J (training data)

    Parameters
    ----------
    J : np.array
        joint configurations

    Returns
    -------
    F : np.array
        posture features
    """
    F = None
    if os.path.exists(path=config.path_F):
        F = load_numpy(file_path=config.path_F)
    
    if F is None or F.shape[-1] != config.r:
        hnne = HNNE(dim=config.r, ann_threshold=config.num_neighbors)
        # maximum number of data for hnne (11M), we use max_num_data_hnne to test
        num_data = min(config.max_num_data_hnne, len(J))
        F = hnne.fit_transform(X=J[:num_data], dim=config.r, verbose=True)
        # query nearest neighbors for the rest of J
        if len(F) != len(J):
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(J[:num_data])
            neigh_idx = knn.kneighbors(J[num_data:], n_neighbors=1, return_distance=False)
            neigh_idx = neigh_idx.flatten()
            F = np.row_stack((F, F[neigh_idx]))

        save_numpy(file_path=config.path_F, arr=F)
    print(f"F load successfully from {config.path_F}")

    return F


def get_train_loader(J: np.array, P: np.array, F: np.array, batch_size: int = config.batch_size, device: str = config.device):
    """
    a training loader

    :param J: joint configurations
    :type J: np.array
    :param P: end-effector positions
    :type P: np.array
    :param F: posture features
    :type F: np.array
    :return: torch dataloader
    :rtype: dataloader
    """
    assert len(J) == len(P) and len(P) == len(F)

    C = np.column_stack((P, F))

    dataset = create_dataset(features=J, targets=C, device=device)
    loader = dataset.create_loader(shuffle=True, batch_size=batch_size)

    return loader


def get_test_loader(P: np.array, F: np.array):
    """
    a testing loader

    :param J: joint configurations
    :type J: np.array
    :param P: end-effector positions
    :type P: np.array
    :param F: posture features
    :type F: np.array
    :return: torch dataloader
    :rtype: dataloader
    """
    assert len(P) < len(F)  # P from testing dataset, F from training dataset
    # Algo 1. random pick up f from F # 0.008 m
    rng = np.random.default_rng()
    rand = rng.integers(low=0, high=len(F), size=len(P))
    f_extended = F[rand]

    # Algo 2. random number f # 0.01 m
    # f_mean = np.repeat(np.atleast_2d(np.mean(F, axis=0)), [len(P)], axis=0)
    # f_rand = np.random.rand(len(P), F.shape[-1]) # [0, 1)
    # f_extended = f_mean + f_rand  * 0.03

    # Algo 3. knn search nearest P (nP), pickup its F

    C = np.column_stack((P, f_extended))
    
    dataset = create_dataset(features=np.zeros_like(C), targets=C)
    loader = dataset.create_loader(shuffle=True, batch_size=config.batch_size)

    return loader


def get_inference_loader(P: np.array, F: np.array, knn=None):
    """
    an inference loader

    Parameters
    ----------
    P : np.array
        end-effector positions for inference
    F : np.array
        posture features from training data
    knn : _type_, optional
        knn model fit P_train, by default None

    Returns
    -------
    inference_loader
        a torch dataloader
    """
    assert len(P) < len(F)

    # get knn
    if knn is None:
        knn = load_pickle(file_path=config.path_knn)

    # get reference F
    f_extended = nearest_neighbor_F(knn, P_ts=P, F=F)

    # create loader
    C = np.column_stack((P, f_extended))
    
    dataset = create_dataset(features=np.zeros_like(C), targets=C)
    loader = dataset.create_loader(shuffle=True, batch_size=config.batch_size)

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


def save_numpy(file_path: str, arr: np.array):
    """
    save arr as a numpy file

    Parameters
    ----------
    file_path : str
        file_path = file_name.np
    arr : np.array
        data
    """

    dir_path = os.path.dirname(p=file_path)
    if not os.path.exists(path=dir_path):
        print(f"mkdir {dir_path}")
        os.mkdir(path=dir_path)
    np.save(file_path, arr)


def add_noise(batch, esp: float = config.noise_esp, step: int = 0, eval: bool = False):
    J, C = batch
    if eval or step < config.num_steps_add_noise:
        std = torch.zeros((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, std))
    else:
        s = torch.rand((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, s))
        noise = torch.normal(
            mean=torch.zeros_like(input=J),
            std=esp * torch.repeat_interleave(input=s, repeats=J.shape[1], dim=1),
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

def train_step(model, batch, optimizer, scheduler):
    """
    _summary_

    Args:
        model (_type_): _description_
        batch (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_

    Returns:
        _type_: _description_
    """
    x, y = add_noise(batch)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()

def data_preprocess_for_inference(P, F, knn):
    # Data Preprocessing: Posture Feature Extraction
    ref_F = nearest_neighbor_F(knn, np.atleast_2d(P), F) # knn
    # ref_F = rand_F(P, F) # f_rand
    # ref_F = pick_F(P, F) # f_pick
    
    C = np.column_stack((np.atleast_2d(P), np.atleast_2d(ref_F)))
    C = C.astype(np.float32)

    # Project to Tensor(device)
    C = torch.from_numpy(C).to(config.device)
    _, C = add_noise((torch.zeros_like(C), C), eval=True)

    return C


def nearest_neighbor_F(knn: NearestNeighbors, P_ts: np.array, F: np.array, n_neighbors: int=1):
    P_ts = np.atleast_2d(P_ts)
    assert len(P_ts) < len(F)
    neigh_idx = knn.kneighbors(P_ts[:, :3], n_neighbors=n_neighbors, return_distance=False)
    neigh_idx = neigh_idx.flatten()
    
    return F[neigh_idx]

def rand_F(P_ts: np.array, F: np.array):
    return np.random.rand(len(np.atleast_2d(P_ts)), F.shape[-1])

def pick_F(P_ts: np.array, F: np.array):
    idx = np.random.randint(low=0, high=len(F), size=len(np.atleast_2d(P_ts)))
    return F[idx]


def inference(
    robot, P_inf: np.array, F: np.array, solver, knn, K: int, print_report: bool = False
):
    """
    inference function, Note that: inference time include data preprocessing and postprocessing.

    Parameters
    ----------
    robot : Robot
        Robot arm
    P_inf : np.array
        end-effector positions for inference
    F : np.array
        posture features from training data
    solver : Normalizing Flow model
        a trained normalizing flow model
    knn : NearestNeighbor
        a fitted knn over P_train
    K : int
        #samples per a position (task point)
    print_report: bool
        print summary of position errors and inference time. Note that it will not return anything.

    Returns
    -------
    J_hat : np.array
        joint configurations
    position_errors : np.array
        position errors
    inference_time : np.array
        inference time of K samples for each P in P_inf
    """
    assert len(P_inf) < 1000
    raise NotImplementedError("Not implement m=7 case")
    position_errors = np.zeros(shape=(len(P_inf), K))
    inference_time = np.zeros(shape=(len(P_inf)))

    for i, P in enumerate(P_inf):
        time_begin = time.time()

        # Data Preprocessing
        C = data_preprocess_for_inference(P=P, F=F, knn=knn)

        # Begin inference
        with torch.inference_mode():
            J_hat = solver(C).sample((K,))
            J_hat = J_hat.detach().cpu().numpy()
        # Calculate Position Errors and Inference Time
        position_errors[i] = robot.position_errors_Arr_Inputs(qs=J_hat, ee_pos=P)
        inference_time[i] = round(time.time() - time_begin, 2)
    position_errors = position_errors.flatten()

    if print_report:
        df = pd.DataFrame(position_errors, columns=["position errors (m)"])
        print(df.describe())
        df = pd.DataFrame(
            inference_time, columns=[f"inference time (sec) of {K} samples"]
        )
        print(df.describe())
    else:
        return J_hat, position_errors, inference_time


def test(
    robot, P_ts: np.array, F: np.array, solver, knn, K: int, print_report: bool = True
):
    """
    test function, Note that: inference time refers to solver inference time, not include data preprocessing or postprocessing.

    Parameters
    ----------
    robot : Robot
        Robot arm
    P_ts : np.array
        end-effector positions for testing
    F : np.array
        posture features from training data
    solver : Normalizing Flow model
        a trained normalizing flow model
    knn : NearestNeighbor
        a fitted knn over P_train
    K : int
        #samples per a position (task point)
    print_report: bool
        print summary of position errors and inference time. Note that it will not return anything.

    Returns
    -------
    J_hat : np.array
        joint configurations
    position_errors : np.array
        position errors
    avg_inference_time : float
        average inference time over #(len(P_ts)*K) samples
    """
    assert len(P_ts) < 1000

    position_errors = np.zeros(shape=(len(P_ts), K))
    orientation_errors = np.zeros(shape=(len(P_ts), K))
    

    # Data Preprocessing
    C = data_preprocess_for_inference(P=P_ts, F=F, knn=knn)

    time_begin = time.time()
    # Begin inference
    with torch.inference_mode():
        J_hat = solver(C).sample((K,))
        J_hat = J_hat.detach().cpu().numpy()

    avg_inference_time = round((time.time() - time_begin) / len(P_ts), 2)

    if config.m == 3:
        # Calculate Position Errors and Inference Time
        for i, P in enumerate(P_ts):
            position_errors[i] = robot.position_errors_Arr_Inputs(
                qs=J_hat[:, i, :], ee_pos=P
            )
        position_errors = position_errors.flatten()

        if print_report:
            df = pd.DataFrame(position_errors, columns=["position errors (m)"])
            print(df.describe())
            print(f"average inference time (of {len(P_ts)} P): {avg_inference_time} sec.")
            return df
        else:
            return position_errors, avg_inference_time
    else:
        # Calculate Position Errors and Inference Time
        for i, P in enumerate(P_ts):
            position_errors[i], orientation_errors[i] = robot.position_orientation_errors_Arr_Inputs(
                qs=J_hat[:, i, :], ee_pos=P
            )
        position_errors = position_errors.flatten()
        orientation_errors = orientation_errors.flatten()
        
        if print_report:
            df = pd.DataFrame([position_errors, orientation_errors], columns=["position errors (m)", "orientation_errors (rad)"])
            print(df.describe())
            print(f"average inference time (of {len(P_ts)} P): {avg_inference_time} sec.")
            return df
        else:
            return position_errors, orientation_errors, avg_inference_time


def sample_J_traj(P_path, ref_F, solver, robot):
    """
    _summary_
    example of use:
    df, qs = sample_J_traj(ee_traj, ref_F, nflow)
    :param P_path: _description_
    :type P_path: _type_
    :param ref_F: _description_
    :type ref_F: _type_
    :param solver: _description_
    :type solver: _type_
    :return: _description_
    :rtype: _type_
    """
    ref_F = np.tile(ref_F, (len(P_path), 1))

    C = np.column_stack((P_path, ref_F, np.zeros((len(P_path),))))
    C = torch.tensor(data=C, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        J_hat = solver(C).sample((1,))

    J_hat = J_hat.detach().cpu().numpy()[0]
    df = eval_J_traj(robot, J_hat, P_path=P_path, position_errors=None)
    return df, J_hat


def sample_P_path(robot, load_time: str = "") -> str:
    """
    _summary_
    example of use

    for generate
    traj_dir = sample_P_path(robot=panda, load_time=')

    for demo
    traj_dir = sample_P_path(robot=panda, load_time='05232300')

    :param robot: _description_
    :type robot: _type_
    :param load_time: _description_, defaults to ''
    :type load_time: str, optional
    :return: _description_
    :rtype: str
    """
    if load_time == "":
        traj_dir = config.traj_dir + datetime.now().strftime("%m%d%H%M%S") + "/"
    else:
        traj_dir = config.traj_dir + load_time + "/"

    P_path_file_path = traj_dir + "ee_traj.npy"
    J_traj_file_path = traj_dir + "q_traj.npy"

    if load_time == "" or not os.path.exists(path=J_traj_file_path):
        # P_path, J_traj = robot.path_generate_via_stable_joint_traj(dist_ratio=0.9, t=20)
        P_path, J_traj = robot.path_generate_via_stable_joint_traj_quaternion(dist_ratio=0.9, t=20)
        
        save_numpy(file_path=P_path_file_path, arr=P_path)
        save_numpy(file_path=J_traj_file_path, arr=J_traj)

    if os.path.exists(path=traj_dir):
        print(f"{traj_dir} load successfully.")

    return traj_dir


def path_following(
    robot,
    Path_dir: str,
    model,
    knn,
    F,
    num_traj: int = 3,
) -> None:
    """
    path following generation for our method
    
    Parameters
    ----------
    robot : _type_
        robotic arm
    Path_dir : str
        path to ee Path, generated by sample_P_path
    model : _type_
        flow, iflow, or nflow
    knn : _type_
        knn of P_train
    F : _type_
        F_train
    num_traj : int, optional
        the number of generated joint trajectory samples, by default 3
    """
    raise NotImplementedError("Need to consider normalize :)")
    def load_and_plot(exp_traj_path: str, ee_path: np.array):
        if os.path.exists(path=exp_traj_path):
            robot.plot(qs=qs)
        else:
            print(f"{exp_traj_path} does not exist !")

    Path = load_numpy(file_path=Path_dir + "ee_traj.npy")
    Path = Path[:, :config.m]

    ref_F = nearest_neighbor_F(knn, np.atleast_2d(Path), F) # knn
    
    exp_path = lambda idx: Path_dir + f"exp_{idx}.npy"
    
    rand_idxs = np.random.randint(low=0, high=len(ref_F), size=num_traj)
    
    for i, rand in enumerate(rand_idxs):
        df, qs = sample_J_traj(Path, ref_F[0], model, robot)
        print(df.describe())
        save_numpy(file_path=exp_path(i), arr=qs)

    for i in range(num_traj):
        load_and_plot(exp_traj_path=exp_path(i), ee_path=Path)


def calc_ang_errs(qs):
    """
    calcuate the sum of difference for angles

    :param qs: _description_
    :type qs: _type_
    :return: _description_
    :rtype: _type_
    """
    ang_errs = np.zeros_like(qs)
    ang_errs[1:] = np.abs(np.diff(qs, axis=0))
    ang_errs[0] = ang_errs[1:].mean(axis=0)
    ang_errs = ang_errs.sum(axis=1)
    ang_errs = np.rad2deg(ang_errs)
    return ang_errs


def eval_J_traj(
    robot, J_traj: np.array, P_path: np.array = None, position_errors: np.array = None
):
    """
    evalution of J_traj for path-following tasks

    Parameters
    ----------
    robot : Robot
        robot arm
    J_traj : np.array
        a joint trajectory
    P_path : np.array, optional
        an end-effector position path, by default None
    position_errors : np.array, optional
        position errors for FK(J_traj)-P_path, by default None

    Returns
    -------
    df : pd.DataFrame
        position errors and ang_errs(sum)
    """
    assert not (P_path is None and position_errors is None)

    ang_errs = calc_ang_errs(qs=J_traj)

    if position_errors is None:
        position_errors = robot.position_errors_Arr_Inputs(qs=J_traj, ee_pos=P_path)

    df = pd.DataFrame(
        np.column_stack((position_errors, ang_errs)),
        columns=["position_errors", "ang_errs(sum)"],
    )
    return df


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