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
from common.config import ConfigFile, ConfigIKP, ConfigDiversity, ConfigPosture
from common.file import load_poses_and_numerical_ik_sols


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
    model_checkpoint = ConfigFile.nodeik_model_path


def evaluate_pose_errors_P2d_P2d(P_hat, P):
    assert P.shape == P_hat.shape
    l2 = np.linalg.norm(P[:, :3] - P_hat[:, :3], axis=1)
    a_quats = np.array([Quaternion(array=a[3:]) for a in P])
    b_quats = np.array([Quaternion(array=b[3:]) for b in P_hat])
    ang = np.array([Quaternion.distance(a, b) for a, b in zip(a_quats, b_quats)])
    return l2, ang


def init_nodeik(args, std: float, robot: Robot=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    wp.init()
    if robot is None:
        robot = Robot(robot_path=ConfigFile.nodeik_urdf_path, ee_link_name="panda_hand")
    learn = Learner.load_from_checkpoint(
        args.model_checkpoint,
        model=build_model(args, robot.active_joint_dim, condition_dims=7).to(device),
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
    x = robot.get_pair()[robot.active_joint_dim :]
    J = np.empty((num_poses, robot.active_joint_dim))
    P = np.empty((num_poses, len(x)))
    for i in trange(num_poses):
        J[i] = robot.get_pair()[: robot.active_joint_dim]
        P[i] = robot.get_pair()[robot.active_joint_dim :]
    return J, P


def ikp():
    config = ConfigIKP()
    robot, nodeik = init_nodeik(args, config.std)

    _, P = get_pair_from_robot(robot, config.num_poses)
    P = np.repeat(
        np.expand_dims(P, axis=1), config.num_sols, axis=1
    )  # (config.num_poses, config.num_sols, len(x))

    begin = time.time()
    J_hat = np.empty((config.num_poses, config.num_sols, robot.active_joint_dim))
    P_hat = np.empty_like(P)
    for i in trange(config.num_poses):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    avg_inference_time = round((time.time() - begin) / config.num_poses, 3)
    l2, ang = evaluate_pose_errors_P2d_P2d(
        P_hat.reshape(-1, P.shape[-1]), P.reshape(-1, P.shape[-1])
    )
    display_ikp(l2.mean(), ang.mean(), avg_inference_time)


def posture_constraint_ikp():
    config = ConfigPosture()
    robot, nodeik = init_nodeik(args, config.std)
    J, P = get_pair_from_robot(robot, config.num_poses)

    # P.shape = (config.num_sols, config.num_poses, len(x))
    P = np.repeat(np.expand_dims(P, axis=0), config.num_sols, axis=0)

    J_hat = np.empty((config.num_sols, config.num_poses, robot.active_joint_dim))
    P_hat = np.empty_like(P)
    for i in trange(config.num_sols):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    l2, ang = evaluate_pose_errors_P2d_P2d(
        P_hat.reshape(-1, P.shape[-1]), P.reshape(-1, P.shape[-1])
    )

    # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # J.shape = (num_poses, num_dofs or n)
    distance_J = compute_distance_J(J_hat, J)
    display_posture(
        config.record_dir,
        "nodeik",
        l2.flatten(),
        ang.flatten(),
        distance_J.flatten(),
        config.success_distance_thresholds,
    )


def load_poses_and_numerical_ik_sols_nodeik(record_dir: str, nodeik: ModelWrapper):
    P, J = load_poses_and_numerical_ik_sols(record_dir)
    print(f"P.shape: {P.shape}, J.shape: {J.shape}")
    P_hat = np.empty_like(P)
    for i in range(len(P)):
        P_hat[i] = nodeik.forward_kinematics(J[i, np.random.randint(0 , J.shape[1])])
    l2, ang = evaluate_pose_errors_P2d_P2d(P_hat, P)
    df = pd.DataFrame({"l2": l2, "ang": ang})
    assert df["l2"].mean() < 1e-3, f"[LOAD ERROR] l2.mean(): {df['l2'].mean()}" 
    # check if the numerical ik solutions are correct
    return P, J


def diversity():
    config = ConfigDiversity()
    robot, nodeik = init_nodeik(args, 0.1)
    P, J = load_poses_and_numerical_ik_sols_nodeik(config.record_dir, nodeik)

    num_poses, num_sols = J.shape[0:2]
    base_stds = config.base_stds

    print(f"num_poses: {num_poses}, num_sols: {num_sols}")

    l2_nodeik = np.empty((len(base_stds)))
    ang_nodeik = np.empty((len(base_stds)))
    mmd_nodeik = np.empty((len(base_stds)))

    # (num_poses, num_sols, len(x))
    P_repeat = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
    assert P_repeat.shape == (num_poses, num_sols, P.shape[-1])

    J_hat_nodeik = np.empty(
        (len(base_stds), num_poses, num_sols, robot.active_joint_dim)
    )
    for i, std in enumerate(base_stds):
        _, nodeik = init_nodeik(args, std, robot)

        J_hat = np.empty_like(J)
        P_hat = np.empty_like(P_repeat)
        for ip in (pbar := trange(num_poses)):
            pbar.set_description(f"i: {i}, std: {round(std, 1)}")
            J_hat[ip], _ = nodeik.inverse_kinematics(P_repeat[ip])
            for isols in range(num_sols):
                P_hat[ip, isols] = nodeik.forward_kinematics(J_hat[ip, isols])
        J_hat = J_hat.reshape(-1, J_hat.shape[-1])
        P_hat = P_hat.reshape(-1, P_hat.shape[-1])
        l2, ang = evaluate_pose_errors_P2d_P2d(
            P_hat, P_repeat.reshape(-1, P_repeat.shape[-1])
        )

        # filter out the outliers
        l2_threshold, ang_threshold = config.pose_error_threshold
        condition = (l2 < l2_threshold) & (np.rad2deg(ang) < ang_threshold)
        l2_nodeik[i] = l2[condition].mean()
        ang_nodeik[i] = ang[condition].mean()
        # print(f"nan condition sum: {condition.sum()}, remaining: {len(condition) - condition.sum()}")
        J_hat[~condition] = np.nan
        P_hat[~condition] = np.nan

        J_hat = J_hat.reshape(num_poses, num_sols, J_hat.shape[-1])
        P_hat = P_hat.reshape(num_poses, num_sols, P_hat.shape[-1])

        J_hat_nodeik[i] = J_hat
        mmd_nodeik[i] = mmd_evaluate_multiple_poses(J_hat, J, num_poses)

    save_diversity(
        config.record_dir,
        "nodeik",
        J_hat_nodeik,
        l2_nodeik,
        ang_nodeik,
        mmd_nodeik,
        base_stds,
    )


if __name__ == "__main__":
    # ikp()
    # posture_constraint_ikp()
    diversity()
