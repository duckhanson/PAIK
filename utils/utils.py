import os
# from tqdm.auto import tqdm
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
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
        return config.path_J_train, config.path_P_train, config.path_F
    else:
        return config.path_J_test, config.path_J_test, config.path_F_test


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
    path_J, path_P, path_F = _get_data_path(len_data=N)

    J = load_numpy(file_path=path_J)
    P = load_numpy(file_path=path_P)

    if len(J) != N:
        J, P = robot.uniform_sample_J(num_samples=N, return_ee=True)
        save_numpy(file_path=path_J, arr=J)
        save_numpy(file_path=path_P, arr=P)

    return J, P



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
    suc_load = False
    path_hnne = config.path_hnne
    if os.path.exists(path=path_hnne):
        try:
            hnne = HNNE.load(path=path_hnne)
            print(f"hnne load successfully from {path_hnne}")
            F = load_numpy(file_path=config.path_F)
            suc_load = True
        except Exception:
            print("hnne load err, assuming you use different architecture.")

    if not suc_load:
        hnne = HNNE(dim=config.r, ann_threshold=config.num_neighbors)
        F = hnne.fit_transform(X=J, dim=config.r, verbose=True)
        hnne.save(path=path_hnne)
        save_numpy(file_path=config.path_F, arr=F)

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

def nearest_neighbor_F(neigh: NearestNeighbors, P_ts: np.array, F: np.array):
    assert len(P_ts) < len(F)

    neigh_idx = neigh.kneighbors(P_ts, n_neighbors=1, return_distance=False)
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

    # Data Preprocessing
    C = data_preprocess_for_inference(P=P_ts, F=F, knn=knn)

    time_begin = time.time()
    # Begin inference
    with torch.inference_mode():
        J_hat = solver(C).sample((K,))
        J_hat = J_hat.detach().cpu().numpy()

    avg_inference_time = round((time.time() - time_begin) / len(P_ts), 2)

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
        return J_hat, position_errors, avg_inference_time


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
    traj_dir: str,
    hnne=None,
    model=None,
    num_traj: int = 3,
    enable_regenerate: bool = False,
) -> None:
    # TODO
    raise NotImplementedError("need to remove hnne and the following implemenation")
    pass
    # """
    # _summary_
    #     for generate
    #     path_following(hnne=hnne, num_traj=3, model=nflow, robot=panda, traj_dir=traj_dir)

    #     only for demo
    #     path_following(hnne=None, num_traj=3, model=None, robot=panda, traj_dir=traj_dir)

    # :param robot: _description_
    # :type robot: _type_
    # :param traj_dir: _description_
    # :type traj_dir: str
    # :param hnne: _description_, defaults to None
    # :type hnne: _type_, optional
    # :param model: _description_, defaults to None
    # :type model: _type_, optional
    # :param num_traj: _description_, defaults to 3
    # :type num_traj: int, optional
    # :param enable_regenerate: _description_, defaults to False
    # :type enable_regenerate: bool, optional
    # """

    # def load_and_plot(exp_traj_path: str, ee_path: np.array):
    #     if os.path.exists(path=exp_traj_path):
    #         err = np.zeros((100,))
    #         qs = load_numpy(file_path=exp_traj_path)
    #         for i in range(len(qs)):
    #             err[i] = robot.position_errors_Single_Input(q=qs[i], ee_pos=ee_traj[i])
    #         print(qs)
    #         robot.plot(qs=qs)
    #     else:
    #         print(f"{exp_traj_path} does not exist !")

    # already_exists = True

    # def exp_path(idx):
    #     return traj_dir + f"exp_{i}.npy"

    # for i in range(num_traj):
    #     if not os.path.exists(path=exp_path(i)):
    #         already_exists = False
    #         break

    # ee_traj = load_numpy(file_path=traj_dir + "ee_traj.npy")
    # q_traj = load_numpy(file_path=traj_dir + "q_traj.npy")

    # if (
    #     enable_regenerate
    #     or not already_exists
    #     and hnne is not None
    #     and model is not None
    # ):
    #     rand = np.random.randint(low=0, high=len(q_traj), size=num_traj)
    #     pidx = hnne.transform(X=q_traj[rand])
    #     print(pidx)

    #     for i, ref_F in enumerate(pidx):
    #         df, qs = sample_J_traj(ee_traj, ref_F, model, robot)
    #         print(df.describe())
    #         save_numpy(file_path=exp_path(i), arr=qs)

    # for i in range(num_traj):
    #     load_and_plot(exp_traj_path=exp_path(i), ee_path=ee_traj)


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
