from dataclasses import dataclass

import os
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
from sklearn.cluster import AgglomerativeClustering

def cluster_based_on_distance(a, dist_thresh=1):
    kmeans= AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh).fit(a)
    return a[np.sort(np.unique(kmeans.labels_, return_index=True)[1])]

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
    model_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'model','panda_loss-20.ckpt')
    
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
    r = Robot(robot_path=os.path.join(os.path.dirname(__file__), 'assets', 'robots','franka_panda', 'panda_arm.urdf'), ee_link_name='panda_hand')
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
    r = Robot(robot_path=os.path.join(os.path.dirname(__file__), 'assets', 'robots','franka_panda', 'panda_arm.urdf'), ee_link_name='panda_hand')
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
    

def evaluate_diversity():
    NUM_POSES = 10_000
    N_NEIGHBORS = 500
    NUM_SOLS = N_NEIGHBORS
    LAMBDA = (0.005, 0.05)
    STD = 0.25

    num_poses = NUM_POSES
    num_sols = NUM_SOLS
    n_neighbors = N_NEIGHBORS
    verbose = True
    batch_size = 5000
    lambda_ = LAMBDA
    joint_cofig_distance_thres_rads = 2
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    r = Robot(robot_path=os.path.join(os.path.dirname(__file__), 'assets', 'robots','franka_panda', 'panda_arm.urdf'), ee_link_name='panda_hand')
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
    # P.shape = (num_poses, num_sols, len(x))
    print('P:', P.shape)
    
    J_hat = np.empty((num_poses, num_sols, r.active_joint_dim))
    P_hat = np.empty((num_poses, num_sols, len(x)))
    
    if num_poses < num_sols:
        for i in trange(num_poses):
            J_hat[i], _ = nodeik.inverse_kinematics(P[i])
            P_hat[i] = nodeik.forward_kinematics(J_hat[i])
    else:
        for i in trange(num_sols):
            J_hat[:, i, :], _ = nodeik.inverse_kinematics(P[:, i, :])
            P_hat[:, i, :] = nodeik.forward_kinematics(J_hat[:, i, :])
    l2, ang = evalutate_pose_errors(J_hat.reshape(-1, r.active_joint_dim), P_hat.reshape(-1, len(x)), P.reshape(-1, len(x))) 
    df = pd.DataFrame({
        'l2': l2,
        'ang (rad)': ang,
        'ang (deg)': np.rad2deg(ang),
    })
    print(df.describe())
    
    
    J_hat = J_hat.reshape(num_poses, num_sols, -1)
    l2 = l2.reshape(num_poses, num_sols)
    ang = ang.reshape(num_poses, num_sols)
    
    n_clusters = np.empty((num_poses))
    for i in trange(num_poses):
        # print("Filter by l2 and ang")
        J_candidate = J_hat[i][(l2[i] < lambda_[0]) & (ang[i] < lambda_[1])]
        # print(J_candidate.shape)
        # print("J_candidate.shape", J_candidate.shape)
        # print(f"Cluster by joint distance")
        if J_candidate.shape[0] < 2:
            n_clusters[i] = J_candidate.shape[0]    
        else:
            J_filtered = cluster_based_on_distance(J_candidate, dist_thresh=joint_cofig_distance_thres_rads)
            # print("J_filtered.shape", J_filtered.shape)
            n_clusters[i] = J_filtered.shape[0]
        
    df = pd.DataFrame({
        'n_clusters': n_clusters
    })
    print(df.describe())
    
    
if __name__ == '__main__':
    # run_random(NUM_POSES, NUM_SOLS)
    # run_path_following(LOAD_TRAJ)
    evaluate_diversity()    
    
    