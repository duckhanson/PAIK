from dataclasses import dataclass

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

PAFIK_WORKDIR = "/home/luca/pafik"
WORK_DIR = "/home/luca/nodeik"
NUM_POSES = 30
NUM_SOLS = 2000
BATCH_SIZE = 5000
BASE_STDS = np.arange(0.1, 1.5, 0.1)  # start, stop, step
STD = 0.01


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
    device = torch.device("cuda:" + str(args.gpu)
                          if torch.cuda.is_available() else "cpu")

    wp.init()
    if robot is None:
        robot_urdf_path = WORK_DIR + "/examples/assets/robots/franka_panda/panda_arm.urdf"
        robot = Robot(robot_path=robot_urdf_path, ee_link_name="panda_hand")
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
    for i in range(num_poses):
        J[i] = robot.get_pair()[: robot.active_joint_dim]
        P[i] = robot.get_pair()[robot.active_joint_dim:]
    return J, P


def ikp(num_poses, num_sols):
    robot, nodeik = init_nodeik(args, STD)

    _, P = get_pair_from_robot(robot, num_poses)
    P = np.repeat(np.expand_dims(P, axis=1), num_sols,
                  axis=1)  # (num_poses, num_sols, len(x))

    begin = time.time()
    J_hat = np.empty((num_poses, num_sols, robot.active_joint_dim))
    P_hat = np.empty_like(P)
    for i in trange(num_poses):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    avg_inference_time = round((time.time() - begin) / num_poses, 3)
    l2, ang = evalutate_pose_errors_Phat2d_P2d(
        P_hat.reshape(-1, P.shape[-1]),
        P.reshape(-1, P.shape[-1]),
    )
    df = pd.DataFrame(
        {
            "l2": l2,
            "ang (deg)": np.rad2deg(ang),
        }
    )
    print(df.describe())
    print(f"avg_inference_time: {avg_inference_time}")


def load_poses_and_numerical_ik_sols(date: str, nodeik: ModelWrapper):
    P = np.load(f"{PAFIK_WORKDIR}/record/{date}/poses.npy")
    J = np.load(f"{PAFIK_WORKDIR}/record/{date}/numerical_ik_sols.npy")
    P_hat = np.empty_like(P)
    for i in trange(len(P)):
        P_hat[i] = nodeik.forward_kinematics(J[i, np.random.randint(0, J.shape[1])])
    l2, ang = evalutate_pose_errors_Phat2d_P2d(P_hat, P)
    assert l2.mean() < 1e-3  # check if the numerical ik solutions are correct
    return P, J


def mmd_posture_diversity():
    robot, nodeik = init_nodeik(args, STD)
    P, J = load_poses_and_numerical_ik_sols("2024_02_22", nodeik)

    num_poses, num_sols = J.shape[0: 2]
    base_stds = BASE_STDS

    print(f"num_poses: {num_poses}, num_sols: {num_sols}")
    
    l2_nodeik = np.empty((len(base_stds)))
    ang_nodeik = np.empty((len(base_stds)))
    mmd_nodeik = np.empty((len(base_stds)))

    P_repeat = np.repeat(np.expand_dims(P, axis=1), num_sols, axis=1) # (num_poses, num_sols, len(x))
    assert P_repeat.shape == (num_poses, num_sols, P.shape[-1])
    
    for i, base_std in enumerate(base_stds):
        _, nodeik = init_nodeik(args, base_std, robot)
        
        J_hat = np.empty_like(J)
        P_hat = np.empty_like(P_repeat)
        for ip in trange(num_poses):
            J_hat[ip], _ = nodeik.inverse_kinematics(P_repeat[ip])
            for isols in range(num_sols):
                P_hat[ip, isols] = nodeik.forward_kinematics(J_hat[ip, isols])
        l2, ang = evalutate_pose_errors_Phat2d_P2d(P_hat.reshape(-1, P_hat.shape[-1]), P_repeat.reshape(-1, P_repeat.shape[-1]))
        print(f"base_std: {base_std}, l2: {l2.mean()}, ang: {np.rad2deg(ang.mean())}")
        break
        
            

    
    # for _ in trange(num_poses // batch_size):

        # J_hat = np.empty((batch_size, num_sols, robot.active_joint_dim))
        # P_hat = np.empty((batch_size, num_sols, len(x)))

        # for i in range(batch_size):
        #     J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        #     P_hat[i] = nodeik.forward_kinematics(J_hat[i])
        # l2, ang = evalutate_pose_errors(
        #     J_hat.reshape(-1, robot.active_joint_dim),
        #     P_hat.reshape(-1, len(x)),
        #     P.reshape(-1, len(x)),
        # )

        # J_hat = J_hat.reshape(batch_size, num_sols, -1)
        # l2 = l2.reshape(batch_size, num_sols)
        # ang = ang.reshape(batch_size, num_sols)


if __name__ == "__main__":
    
    # ikp(NUM_POSES, NUM_SOLS)
    mmd_posture_diversity()