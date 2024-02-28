from dataclasses import dataclass
import os
import pickle
from datetime import datetime
from tqdm import trange
from tabulate import tabulate
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import time
from nodeik.utils import build_model
import warp as wp
from nodeik.robots.robot import Robot
from nodeik.training import KinematicsDataset, Learner, ModelWrapper
from pyquaternion import Quaternion
from mmd_helper import mmd_evaluate_multiple_poses

paik_WORKDIR = "/home/luca/paik"
WORK_DIR = "/home/luca/nodeik"
URDF_PATH = WORK_DIR + "/examples/assets/robots/franka_panda/panda_arm.urdf"
NUM_POSES = 100
NUM_SOLS = 1000
BATCH_SIZE = 5000
BASE_STDS = np.arange(0.1, 1.5, 0.1)  # start, stop, step
STD = 0.25


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
    model_checkpoint = WORK_DIR + "/model/panda_loss-20.ckpt"


def evalutate_pose_errors_Phat2d_P2d(P_hat, P):
    assert P.shape == P_hat.shape
    l2 = np.linalg.norm(P[:, :3] - P_hat[:, :3], axis=1)
    a_quats = np.array([Quaternion(array=a[3:]) for a in P])
    b_quats = np.array([Quaternion(array=b[3:]) for b in P_hat])
    ang = np.array([Quaternion.distance(a, b)
                   for a, b in zip(a_quats, b_quats)])
    return l2, ang


def init_nodeik(args, std, robot=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    wp.init()
    if robot is None:
        robot = Robot(robot_path=URDF_PATH, ee_link_name="panda_hand")
    learn = Learner.load_from_checkpoint(
        args.model_checkpoint,
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


def ikp(num_poses, num_sols):
    robot, nodeik = init_nodeik(args, STD)

    _, P = get_pair_from_robot(robot, num_poses)
    P = np.repeat(
        np.expand_dims(P, axis=1), num_sols, axis=1
    )  # (num_poses, num_sols, len(x))

    begin = time.time()
    J_hat = np.empty((num_poses, num_sols, robot.active_joint_dim))
    P_hat = np.empty_like(P)
    for i in trange(num_poses):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    avg_inference_time = round((time.time() - begin) / num_poses, 3)
    l2, ang = evalutate_pose_errors_Phat2d_P2d(
        P_hat.reshape(-1, P.shape[-1]), P.reshape(-1, P.shape[-1])
    )
    df = pd.DataFrame({"l2": l2, "ang (deg)": np.rad2deg(ang)})
    print(df.describe())
    print(f"avg_inference_time: {avg_inference_time}")


def load_poses_and_numerical_ik_sols(date: str, nodeik: ModelWrapper):
    record_dir = f"{paik_WORKDIR}/record/{date}"
    P = np.load(f"{record_dir}/poses.npy")
    J = np.load(f"{record_dir}/numerical_ik_sols.npy")
    print(f"loaded from {record_dir}")
    P_hat = np.empty_like(P)
    for i in range(len(P)):
        P_hat[i] = nodeik.forward_kinematics(
            J[i, np.random.randint(0, J.shape[1])])
    l2, ang = evalutate_pose_errors_Phat2d_P2d(P_hat, P)
    assert l2.mean() < 1e-3  # check if the numerical ik solutions are correct
    return P, J


def mmd_posture_diversity(pose_error_threshold=(0.03, 30)):
    robot, nodeik = init_nodeik(args, STD)
    P, J = load_poses_and_numerical_ik_sols(
        datetime.today().strftime('%Y_%m_%d'), nodeik)

    num_poses, num_sols = J.shape[0:2]
    base_stds = BASE_STDS

    print(f"num_poses: {num_poses}, num_sols: {num_sols}")

    l2_nodeik = np.empty((len(base_stds)))
    ang_nodeik = np.empty((len(base_stds)))
    mmd_nodeik = np.empty((len(base_stds)))

    # (num_poses, num_sols, len(x))
    P_repeat = np.repeat(np.expand_dims(P, axis=1), num_sols, axis=1)
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
        l2, ang = evalutate_pose_errors_Phat2d_P2d(
            P_hat, P_repeat.reshape(-1, P_repeat.shape[-1])
        )

        # filter out the outliers
        condition = (l2 < pose_error_threshold[0]) & (
            np.rad2deg(ang) < pose_error_threshold[1]
        )
        l2_nodeik[i] = l2[condition].mean()
        ang_nodeik[i] = ang[condition].mean()
        # print(f"nan condition sum: {condition.sum()}, remaining: {len(condition) - condition.sum()}")
        J_hat[~condition] = np.nan
        P_hat[~condition] = np.nan

        J_hat = J_hat.reshape(num_poses, num_sols, J_hat.shape[-1])
        P_hat = P_hat.reshape(num_poses, num_sols, P_hat.shape[-1])

        J_hat_nodeik[i] = J_hat
        mmd_nodeik[i] = mmd_evaluate_multiple_poses(J_hat, J, num_poses)

    df = pd.DataFrame(
        {
            "l2": l2_nodeik,
            "ang": ang_nodeik,
            "mmd": mmd_nodeik,
            "base_std": base_stds,
        }
    )

    print(df.describe())

    record_dir = f"{paik_WORKDIR}/record/{datetime.today().strftime('%Y_%m_%d')}"
    os.makedirs(record_dir, exist_ok=True)
    np.save(f"{record_dir}/J_hat_nodeik.npy", J_hat_nodeik)
    df.to_pickle(f"{record_dir}/nodeik_posture_mmd_std.pkl")
    print(f"saved to {record_dir}/nodeik_posture_mmd_std.pkl")


if __name__ == "__main__":
    ikp(NUM_POSES, NUM_SOLS)
    # mmd_posture_diversity()
