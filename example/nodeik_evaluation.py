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
from evaluation import n_cluster_analysis, Generate_Diverse_Postures_Info

WORK_DIR = "/home/luca/nodeik"
ROBOT_PATH = WORK_DIR + '/examples/assets/robots/franka_panda/panda_arm.urdf'
MODEL_PATH = WORK_DIR + '/model/panda_loss-20.ckpt'
NUM_POSES = 100
NUM_SOLS = 1000
NUM_STEPS = 20
NUM_TRAJECTORIES = 10000
DDJC_THRES = (40, 80, 120)
STD = 0.01
LOAD_TRAJ = '0201005723'

def max_joint_angle_change(qs: np.ndarray):
    return np.max(np.abs(np.diff(qs, axis=0)))
    

@dataclass
class args:
    layer_type = 'concatsquash'
    dims = '1024-1024-1024-1024'
    num_blocks = 1 
    time_length = 0.5
    train_T = False
    divergence_fn = 'approximate'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    atol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    rtol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    gpu = 0
    rademacher = False
    num_samples = 4
    num_references = 256
    seed = 1
    model_checkpoint = MODEL_PATH
    
np.random.seed(args.seed)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

wp.init()

def evalutate_pose_errors(qs, generated_poses, given_poses, given_qs=None):
    assert qs.shape[0] == given_poses.shape[0] == generated_poses.shape[0]
    assert len(qs.shape) == 2
    l2 = np.linalg.norm(given_poses[:, :3] - generated_poses[:, :3], axis=1)
    a_quats = np.array([Quaternion(array=a[3:]) for a in given_poses])
    b_quats = np.array([Quaternion(array=b[3:]) for b in generated_poses])
    ang = np.array([Quaternion.distance(a, b) for a, b in zip(a_quats, b_quats)])
    if given_qs is not None:
        mjac = max_joint_angle_change(qs)
        ddjc = np.linalg.norm(qs - given_qs, axis=-1)
        return l2, ang, mjac, ddjc
    return l2, ang

def run_random(num_poses, num_sols):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    r = Robot(robot_path=ROBOT_PATH, ee_link_name='panda_hand')
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=build_model(args, r.active_joint_dim, condition_dims=7).to(device), robot=r, std=STD, state_dim=r.active_joint_dim, condition_dim=7)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    nodeik.eval()

    x = r.get_pair()[r.active_joint_dim:]
    P = np.empty((num_poses, len(x)))
    for i in range(num_poses):
        P[i] = r.get_pair()[r.active_joint_dim:]
    P = np.tile(P, (num_sols, 1))
    P = P.reshape(num_poses, num_sols, len(x))
    
    begin = time.time()
    J_hat = np.empty((num_poses, num_sols, r.active_joint_dim))
    P_hat = np.empty((num_poses, num_sols, len(x)))
    for i in trange(num_poses):
        J_hat[i], _ = nodeik.inverse_kinematics(P[i])
        P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    avg_inference_time = round((time.time() - begin) / num_poses, 3)
    l2, ang = evalutate_pose_errors(J_hat.reshape(-1, r.active_joint_dim), P_hat.reshape(-1, len(x)), P.reshape(-1, len(x))) 
    df = pd.DataFrame({
        'l2': l2,
        'ang (deg)': np.rad2deg(ang),
    })
    print(df.describe())
    print(f'avg_inference_time: {avg_inference_time}')
    

def run_path_following(load_time: str):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    r = Robot(robot_path=ROBOT_PATH, ee_link_name='panda_hand')
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=build_model(args, r.active_joint_dim, condition_dims=7).to(device), robot=r, std=STD, state_dim=r.active_joint_dim, condition_dim=7)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    nodeik.eval()
    
    J = np.empty((NUM_TRAJECTORIES, NUM_STEPS, r.active_joint_dim))
    P = np.empty((NUM_TRAJECTORIES, NUM_STEPS, 7))    
    for i in range(NUM_TRAJECTORIES):
        start = r.get_pair()[:r.active_joint_dim]
        stop = r.get_pair()[:r.active_joint_dim]

        J[i] = np.linspace(start, stop, num=NUM_STEPS)
        P[i] = nodeik.forward_kinematics(J[i])
    print('J:', J.shape)
    print('P:', P.shape)
        
    l2 = np.empty((NUM_TRAJECTORIES))
    ang = np.empty((NUM_TRAJECTORIES))
    mjac = np.empty((NUM_TRAJECTORIES))
    ddjc = np.empty((NUM_TRAJECTORIES))
    
    begin = time.time()
    J_hat = np.empty((NUM_TRAJECTORIES, NUM_STEPS, r.active_joint_dim))
    P_hat = np.empty((NUM_TRAJECTORIES, NUM_STEPS, P.shape[-1]))
    
    for i in trange(NUM_STEPS):
        J_hat[:, i, :], _ = nodeik.inverse_kinematics(P[:, i, :])
        P_hat[:, i, :] = nodeik.forward_kinematics(J_hat[:, i, :])
    
    for i in range(NUM_TRAJECTORIES):
        ll, aa, mj, ddj = evalutate_pose_errors(J_hat[i], P_hat[i], P[i], J[i])
        l2[i], ang[i], mjac[i], ddjc[i] = ll.mean(), aa.mean(), mj, ddj.mean()
    
    avg_runtime = (time.time() - begin) / NUM_TRAJECTORIES
    df = pd.DataFrame(
        {
            "l2": l2,
            'ang (deg)': np.rad2deg(ang),
            "mjac (deg)": np.rad2deg(mjac),
            'ddjc (deg)': np.rad2deg(ddjc),
        }
    )
    
    print(
        tabulate(
            [
                (
                    thres,
                    df.query(f"ddjc < {thres}")["ddjc"].count() / df.shape[0],
                )
                for thres in DDJC_THRES
            ],
            headers=["ddjc", "success rate"],
        )
    )
    print(df.describe())
    print(f"avg_inference_time: {avg_runtime}")
    

def posture_diversity():
    NUM_POSES = 5_000
    NUM_SOLS = 15_000  # IKFlow, NODE IK
    STD = 0.25
    SOLUTIONS_SUCCESS_RATE_THRESHOLD_FOR_CLUSTERING_IN_NUM_SOLS = 0.3 # 0.80
    N_CLUSTERS_THRESHOLD = [10, 15, 20, 25, 30]
    
    num_poses = NUM_POSES
    num_sols = NUM_SOLS
    n_clusters_threshold = N_CLUSTERS_THRESHOLD
    success_rate_thresold = SOLUTIONS_SUCCESS_RATE_THRESHOLD_FOR_CLUSTERING_IN_NUM_SOLS
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    r = Robot(robot_path=ROBOT_PATH, ee_link_name='panda_hand')
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=build_model(args, r.active_joint_dim, condition_dims=7).to(device), robot=r, std=STD, state_dim=r.active_joint_dim, condition_dim=7)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    nodeik.eval()
    
    x = r.get_pair()[r.active_joint_dim:]
    P = np.empty((num_poses, len(x)))
    for i in range(num_poses):
        P[i] = r.get_pair()[r.active_joint_dim:]
    P = np.expand_dims(P, axis=1)
    P = np.repeat(P, num_sols, axis=1)
    
    J_hat = np.empty((num_poses, num_sols, r.active_joint_dim))
    P_hat = np.empty((num_poses, num_sols, len(x)))
    
    begin_time = time.time()
    if num_poses < num_sols:
        for i in trange(num_poses):
            J_hat[i], _ = nodeik.inverse_kinematics(P[i])
            P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    else:
        for i in trange(num_sols):
            J_hat[:, i, :], _ = nodeik.inverse_kinematics(P[:, i, :])
            P_hat[:, i, :] = nodeik.forward_kinematics(J_hat[:, i, :])
    l2, ang = evalutate_pose_errors(J_hat.reshape(-1, r.active_joint_dim), P_hat.reshape(-1, len(x)), P.reshape(-1, len(x))) 
    average_time = (time.time() - begin_time) / num_poses
    
    J_hat = J_hat.reshape(num_poses, num_sols, -1)
    l2 = l2.reshape(num_poses, num_sols)
    ang = ang.reshape(num_poses, num_sols)
    
    # ans = n_cluster_analysis(J_hat, l2, ang, num_poses)
    # print(ans)
    df = pd.DataFrame(n_cluster_analysis(J_hat, l2, ang, num_poses, n_clusters_threshold=n_clusters_threshold), columns=n_clusters_threshold)
    print(df.info())
    print(df.describe())
    
    # print a tbulate of the average time to reach n clusters
    N_posture_info = Generate_Diverse_Postures_Info(
        "N", average_time, num_poses, num_sols, df, success_rate_thresold
    )
    
    print(N_posture_info.get_list_of_name_and_average_time_to_reach_n_clusters())
    
    
if __name__ == '__main__':
    # run_random(NUM_POSES, NUM_SOLS)
    # run_path_following(LOAD_TRAJ)
    posture_diversity()    
    
    