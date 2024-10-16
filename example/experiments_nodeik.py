from dataclasses import dataclass
from json import load
from math import isnan
import os
from datetime import datetime
from pickle import NONE
from tqdm import trange
import numpy as np
import pandas as pd
from tqdm import trange
import torch
import time
from common.file import save_diversity
from nodeik.utils import build_model
import warp as wp
from nodeik.robots.robot import Robot
from nodeik.training import Learner, ModelWrapper
from pyquaternion import Quaternion
from common.evaluate import mmd_evaluate_multiple_poses, compute_distance_J
from common.display import display_posture, display_ikp
from common.config import Config_File, Config_IKP, Config_Diversity, Config_Posture
from common.file import load_poses_and_numerical_ik_sols
from jrl.robots import Panda

@dataclass
class args:
    layer_type = "concatsquash"
    dims = "1024-1024-1024-1024"
    num_blocks = 1
    time_length = 0.5
    train_T = False
    divergence_fn = "approximate"
    nonlinearity = "tanh"
    solver = "dopri5"
    atol = 1e-5  # 1e-3 for fast inference, 1e-5 for accurate inference
    rtol = 1e-5  # 1e-3 for fast inference, 1e-5 for accurate inference
    gpu = 0
    rademacher = False
    num_samples = 4
    num_references = 256
    seed = 1
    nodeik_workdir = "/home/luca/nodeik"


def evaluate_pose_errors_P2d_P2d(P_hat, P):
    assert P.shape == P_hat.shape
    l2 = np.linalg.norm(P[:, :3] - P_hat[:, :3], axis=1)
    a_quats = np.array([Quaternion(array=a[3:]) for a in P])
    b_quats = np.array([Quaternion(array=b[3:]) for b in P_hat])
    ang = np.array([Quaternion.distance(a, b)
                   for a, b in zip(a_quats, b_quats)])
    return l2, ang


def init_nodeik(args, std: float, robot: Robot = None):
    config = Config_File()
    config.nodeik_workdir = args.nodeik_workdir

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    wp.init()
    if robot is None:
        robot = Robot(robot_path=Config_File.nodeik_urdf_path,
                      ee_link_name="panda_hand")
    learn = Learner.load_from_checkpoint(
        Config_File.nodeik_model_path,
        model=build_model(args, robot.active_joint_dim,
                          condition_dims=7).to(device),
        robot=robot,
        std=std,
        state_dim=robot.active_joint_dim,
        condition_dim=7,
    )
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    nodeik.eval()
    return robot, nodeik


def get_pair_from_robot(robot, num_poses):
    x = robot.get_pair()[robot.active_joint_dim:]
    J = np.empty((num_poses, robot.active_joint_dim))
    P = np.empty((num_poses, len(x)))
    for i in trange(num_poses):
        J[i] = robot.get_pair()[: robot.active_joint_dim]
        P[i] = robot.get_pair()[robot.active_joint_dim:]
    return J, P

def ikp_numerical_ik_sols(P_num_poses_num_sols, krbt=Panda()):
    num_poses, num_sols = P_num_poses_num_sols.shape[0:2]
    print(f"Start numerical ik solutions for {num_poses} poses and {num_sols} solutions")
    J_hat = np.empty(
        (num_poses, num_sols, robot.active_joint_dim))
    for i in trange(num_poses):
        for j in range(num_sols):
            J_hat[i, j] = krbt.inverse_kinematics_klampt(P_num_poses_num_sols[i, j])
    
    return J_hat


def ikp(config: Config_IKP, robot: Robot, nodeik: ModelWrapper):
    _, P = get_pair_from_robot(robot, config.num_poses)
    P = np.repeat(
        np.expand_dims(P, axis=1), config.num_sols, axis=1
    )  # (config.num_poses, config.num_sols, len(x))

    begin = time.time()
    # J_numerical = ikp_numerical_ik_sols(P, krbt=Panda())
    numerical_inference_time = round((time.time() - begin) / config.num_poses, 3)

    print(f"Start nodeik ik solutions for {config.num_poses} poses and {config.num_sols} solutions")
    begin = time.time()
    J_hat = np.empty(
        (config.num_poses, config.num_sols, robot.active_joint_dim))
    P_hat = np.empty_like(P)
    for i in trange(config.num_poses):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    l2, ang = evaluate_pose_errors_P2d_P2d(
        P_hat.reshape(-1, P.shape[-1]), P.reshape(-1, P.shape[-1])
    )

    # transpose to (num_sols, num_poses, len(x))
    # compute mmd
    # J_hat = J_hat.transpose(1, 0, 2)
    # J_numerical = J_numerical.transpose(1, 0, 2)
    # mmd = mmd_evaluate_multiple_poses(J_hat, J_numerical, config.num_poses)
    avg_inference_time = round((time.time() - begin) / config.num_poses, 3) * 1000

    
    l2_mm = l2[~np.isnan(l2)].mean() * 1000
    ang_deg = np.rad2deg(ang[~np.isnan(ang)].mean())
    
    df = pd.DataFrame({
        "robot": ["panda"] * 2,
        "solver": ["numerical", "nodeik"],
        "l2_mm": [0, l2_mm],
        "ang_deg": [0, ang_deg],
        # "mmd": [0, mmd],
        "inference_time (ms)": [numerical_inference_time, avg_inference_time],
    })
    print(df)
    df.to_csv(f"{config.record_dir}/ikp_panda_nodeik_{config.num_poses}_{config.num_sols}_{config.std}.csv")


if __name__ == "__main__":
    # ikp()
    config = Config_IKP()
    config.num_poses = 500
    robot, nodeik = init_nodeik(args, config.std)
    ikp(config, robot, nodeik)
        
    
    