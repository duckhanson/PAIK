import numpy as np
from datetime import datetime
import os
# from numpy import linalg as LA
from klampt.math import so3
from klampt.model import trajectory

from tqdm import tqdm
from utils.settings import config as cfg
from utils.utils import create_robot_dirs, save_numpy

from jrl.robots import Panda
from jrl.evaluation import pose_errors_cm_deg
import torch


def get_robot(robot_name: str = cfg.robot_name):
    create_robot_dirs()
    
    if robot_name == 'panda':
        return Panda()
    else:
        raise NotImplementedError()

def sample_P_path(load_time: str = "", num_steps=20) -> str:
    """
    _summary_
    example of use

    for generate
    traj_dir = sample_P_path(load_time=')

    for demo
    traj_dir = sample_P_path(load_time='05232300')

    :param robot: _description_
    :type robot: _type_
    :param load_time: _description_, defaults to ''
    :type load_time: str, optional
    :return: _description_
    :rtype: str
    """
    if load_time == "":
        traj_dir = cfg.traj_dir + datetime.now().strftime("%m%d%H%M%S") + "/"
    else:
        traj_dir = cfg.traj_dir + load_time + "/"

    P_path_file_path = traj_dir + "ee_traj.npy"

    if load_time == "" or not os.path.exists(path=P_path_file_path):
        endPoints = np.random.rand(2, cfg.m) # 2 for begin and end
        traj = trajectory.Trajectory(milestones=endPoints)
        P_path = np.empty((num_steps, cfg.m))
        for i in range(num_steps):
            iStep = i/num_steps
            point = traj.eval(iStep)
            P_path[i] = point
        
        save_numpy(file_path=P_path_file_path, arr=P_path)

    if os.path.exists(path=traj_dir):
        print(f"{traj_dir} load successfully.")

    return traj_dir

def sample_J_traj(P_path, ref_F, solver):
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
    assert solver.shrink_ratio < 0.2, "shrink_ratio should be less than 0.2"
    assert solver.shrink_ratio > 0.0, "shrink_ratio should be greater than 0.0"
    P_path = P_path[:, :cfg.m]
    ref_F = np.tile(ref_F, (len(P_path), 1))

    C = np.column_stack((P_path, ref_F, np.zeros((len(P_path),))))
    C = torch.tensor(data=C, device="cuda", dtype=torch.float32)

    J_hat = solver.sample(C, 1)

    J_hat = J_hat.detach().cpu().numpy()[0]
    # df = eval_J_traj(robot, J_hat, P_path=P_path, position_errors=None)
    # return df, J_hat
    return J_hat