from typing import Any, Optional
import numpy as np
from time import time

import pandas as pd
import scipy
from tqdm import tqdm

from common.evaluate import evaluate_pose_error_J3d_P2d
from paik.solver import NSF, PAIK, Solver, get_solver
from ikp import get_robot, numerical_inverse_kinematics_batch, compute_mmd, gaussian_kernel, inverse_multiquadric_kernel

import torch

# set the same random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


from functools import partial
import os
from numpy import ndarray
from tqdm import tqdm, trange
import itertools
from tqdm.contrib import itertools as tqdm_itertools

from paik.file import load_pickle, save_pickle
from latent_space_sampler import Retriever


def solver_batch(solver, poses, num_sols, std=0.001, retriever: Optional[Retriever] = None, J_ref=None, radius=0.0, num_clusters=30, num_seeds_per_pose=10, use_samples=int(5e6), verbose=False, retr_type='cluster'):
    # shape: (num_sols, num_poses, m)
    P_num_sols = np.expand_dims(poses, axis=0).repeat(num_sols, axis=0)
    # shape: (num_sols*num_poses, n)
    P_num_sols = P_num_sols.reshape(-1, poses.shape[-1])
    
    J_ref_num_sols = None
    if J_ref is not None:
        J_ref_num_sols = np.expand_dims(J_ref, axis=0).repeat(num_sols, axis=0)
        J_ref_num_sols = J_ref_num_sols.reshape(-1, J_ref.shape[-1])

    if isinstance(solver, PAIK):
        solver.base_std = std
        F = solver.get_reference_partition_label(P=poses, J=J_ref, num_sols=num_sols)
        # shape: (1, num_sols*num_poses, n)
        J_hat = solver.generate_ik_solutions(P=P_num_sols, F=F, verbose=verbose)
    elif isinstance(solver, NSF):
        if retriever is None:
            solver.base_std = std
            J_hat = solver.generate_ik_solutions(P=poses, num_sols=num_sols)
        else:
            if retr_type == 'cluster':
                latents = retriever.cluster_retriever(seeds=J_ref, num_poses=poses.shape[0], num_sols=num_sols, max_samples=use_samples, radius=radius, n_clusters=num_clusters)
            elif retr_type == 'random':
                latents = retriever.random_retriever(seeds=J_ref, num_poses=poses.shape[0], max_samples=use_samples, num_sols=num_sols, radius=radius)
            elif retr_type == 'numerical':
                latents = retriever.numerical_retriever(poses=poses, seeds=J_ref, num_sols=num_sols, num_seeds_per_pose=num_seeds_per_pose, radius=radius)
            J_hat = solver.generate_ik_solutions(P=P_num_sols, latents=latents, verbose=verbose)
    else:
        J_hat = np.empty((num_sols, poses.shape[0], solver.robot.n_dofs))
        P_torch = torch.tensor(poses, dtype=torch.float32).to('cuda')
        for i, p in enumerate(P_torch):
            solutions = solver.generate_ik_solutions(
                p,
                num_sols,
                latent_distribution='gaussian',
                latent_scale=std,
                clamp_to_joint_limits=False,
            )
            J_hat[:, i] = solutions.detach().cpu().numpy()
    # return shape: (num_sols, num_poses, n)
    return J_hat.reshape(num_sols, poses.shape[0], -1)


def random_ikp(robot, poses: np.ndarray, solve_fn_batch: Any, num_poses: int, num_sols: int, J_hat_num: Optional[np.ndarray] = None):
    begin = time()
    # shape: (num_poses, num_sols, num_dofs or n)
    J_hat = solve_fn_batch(P=poses, num_sols=num_sols)
    assert J_hat.shape == (
        num_sols, num_poses, robot.n_dofs), f"J_hat shape {J_hat.shape} is not correct"

    l2, ang = evaluate_pose_error_J3d_P2d(
        #init(num_sols, num_poses, num_dofs or n)
        robot, J_hat, poses, return_all=True
    )
    
    num_sols_time_ms = round((time() - begin) / len(poses), 3) * 1000
    
    ret_results = {}
    l2_mean = np.nanmean(l2)
    ang_mean = np.nanmean(ang)
    
    ret_results[f'{num_poses}_{num_sols}'] = {
        "l2_mm": l2_mean * 1000,
        "ang_deg": np.rad2deg(ang_mean),
        "num_sols_time_ms": num_sols_time_ms
    }
    
    if J_hat_num is None:
        mmd_guassian = np.nan
        mmd_imq = np.nan
    else:
        mmd_guassian_list = np.empty((num_poses))
        mmd_imq_list = np.empty((num_poses))
        for i in range(num_poses):
            mmd_guassian_list[i] = compute_mmd(J_hat[:, i], J_hat_num[:, i], kernel=gaussian_kernel)
            mmd_imq_list[i] = compute_mmd(J_hat[:, i], J_hat_num[:, i], kernel=inverse_multiquadric_kernel)
        mmd_guassian = mmd_guassian_list.mean()
        mmd_imq = mmd_imq_list.mean()
        
    ret_results[f'{num_poses}_{num_sols}']['mmd_guassian'] = mmd_guassian
    ret_results[f'{num_poses}_{num_sols}']['mmd_imq'] = mmd_imq

    return J_hat, ret_results

def nested_dict_to_2d_dict(nested_dict: dict):
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for k, v in value.items():
                ret_dict[f"{key}_{k}"] = v
        else:
            ret_dict[key] = value
    return ret_dict


def random_ikp_with_mmd(record_dir, robot_name, num_poses, num_sols, paik_std_list, radius_list, num_clusters_list, num_seeds_per_pose_list):
    robot = get_robot(robot_name)
    nsf = get_solver(arch_name="nsf", robot=robot, load=True, work_dir='/home/luca/paik')
    retriever = Retriever(nsf)
    max_samples = int(5e6)
    retriever.init([max_samples], num_clusters_list)
    paik = get_solver(arch_name="paik", robot=robot, load=True, work_dir='/home/luca/paik')
    
    file_path = f"{record_dir}/random_ikp_with_mmd_{robot_name}_{num_poses}_{num_sols}.pkl"
    
    results = {}
    # if os.path.exists(file_path):
    #     results = load_pickle(file_path)
    #     ret_results = nested_dict_to_2d_dict(results)
    #     df = pd.DataFrame(ret_results).T
    #     # round to 4 decimal places
    #     df = df.round(4)
    #     print(df)
    #     print(f"Results are loaded from {file_path}")
    # else:
    #     print(f"Results are not found in {file_path}")
        
    if 'poses' in results:
        poses = results['poses']
    else:
        _, poses = nsf.robot.sample_joint_angles_and_poses(n=num_poses)
        
    print(f"Start numerical IK...")
    # num's variable: num_poses, num_sols
    num_solver_batch = partial(numerical_inverse_kinematics_batch, solver=nsf)    
    J_hat_num, results['num'] = random_ikp(robot, poses, num_solver_batch, num_poses=num_poses, num_sols=num_sols)
    save_pickle(file_path, results)    
    print(f"Results numerical IK are saved in {file_path}")
    
    print(f"Start paik...")
    # paik's variable: num_poses, num_sols, std, 
    for std in tqdm(paik_std_list):
        paik_solver_batch = partial(solver_batch, solver=paik, std=std)
        name = f'paik_gaussian_{std}'
        if name not in results:
            _, results[name] = random_ikp(robot, poses, paik_solver_batch, num_poses=num_poses, num_sols=num_sols, J_hat_num=J_hat_num)
            save_pickle(file_path, results) 
    print(f"Results paik are saved in {file_path}")
    
    print(f"Start nsf w/o retreiver...")
    # nsf's variable: std
    for std in tqdm(paik_std_list):
        nsf_solver_batch = partial(solver_batch, solver=nsf, std=std, retriever=None)
        name = f'nsf_gaussian_{std}'
        if name not in results:
            _, results[name] = random_ikp(robot, poses, nsf_solver_batch, num_poses=num_poses, num_sols=num_sols, J_hat_num=J_hat_num)
            save_pickle(file_path, results)

    print(f"Start nsf with cluster retriever...")    
    # nsf's variable: num_poses, num_sols, max_samples, radius, num_clusters
    use_samples = max_samples
    for radius, num_clusters in tqdm_itertools.product(radius_list, num_clusters_list):
        nsf_solver_batch = partial(solver_batch, solver=nsf, radius=radius, num_clusters=num_clusters, retriever=retriever, use_samples=use_samples, retr_type='cluster')
        name = f'nsf_cluster_{radius}_{num_clusters}'
        if name not in results:
            _, results[name] = random_ikp(robot, poses, nsf_solver_batch, num_poses=num_poses, num_sols=num_sols, J_hat_num=J_hat_num)
            save_pickle(file_path, results)
    print(f"Results nsf with cluster retriever are saved in {file_path}")
    
    print(f"Start nsf with random retriever...")
    # nsf's variable: num_poses, num_sols, max_samples, radius
    for radius, num_clusters in tqdm_itertools.product(radius_list, num_clusters_list):
        use_samples = min(max_samples, num_clusters)
        nsf_solver_batch = partial(solver_batch, solver=nsf, radius=radius, retriever=retriever, use_samples=use_samples, retr_type='random')
        name = f'nsf_random_{radius}_{use_samples}'
        if name not in results:
            _, results[name] = random_ikp(robot, poses, nsf_solver_batch, num_poses=num_poses, num_sols=num_sols, J_hat_num=J_hat_num)
            save_pickle(file_path, results)
            
    print(f"Start nsf with numerical retriever...")
    # nsf's variable: num_poses, num_sols, max_samples, radius, num_seeds_per_pose
    for radius, num_seeds_per_pose in tqdm_itertools.product(radius_list, num_seeds_per_pose_list):
        nsf_solver_batch = partial(solver_batch, solver=nsf, radius=radius, retriever=retriever, num_seeds_per_pose=num_seeds_per_pose, retr_type='numerical', J_ref=None)
        name = f'nsf_numerical_{radius}_{num_seeds_per_pose}'
        if name not in results:
            _, results[name] = random_ikp(robot, poses, nsf_solver_batch, num_poses=num_poses, num_sols=num_sols, J_hat_num=J_hat_num)
            save_pickle(file_path, results)
    
    ret_results = nested_dict_to_2d_dict(results)

    df = pd.DataFrame(ret_results).T
    # round to 4 decimal places
    df = df.round(4)
    print(df)
    file_path = f"{record_dir}/random_ikp_with_mmd_evaluation_results_{robot_name}_{num_poses}_{num_sols}.csv"
    df.to_csv(file_path)
    print(f"Results are saved in {file_path}")
    
from common.config import Config_IKP

if __name__ == "__main__":
    config = Config_IKP()

    config.workdir = '/mnt/d/pads/Documents/paik_store'

    kwarg = {
        'record_dir': config.record_dir,
        'robot_name': 'panda',
        'num_poses': 1000, # 300, 500, 1000
        'num_sols': 1000,  # 300, 500, 1000
        'paik_std_list': [0.001, 0.1, 0.25], # 0.001, 0.1, 0.25, 0.5, 0.7
        'radius_list': [0.001, 0.1, 0.25], # 0, 0.1, 0.3, 0.5, 0.7, 0.9
        'num_clusters_list': [20, 30, 40, 70], # 13, 16, 19, 25, 30, 40
        'num_seeds_per_pose_list': [20, 30, 40, 70] # 5, 10, 15, 20	
    }

    robot_names = ["panda"] # "panda", "fetch", "fetch_arm", "atlas_arm", "atlas_waist_arm", "baxter_arm"

    for robot_name in robot_names:
        print(f"Start to evaluate {robot_name}...")
        kwarg['robot_name'] = robot_name
        random_ikp_with_mmd(**kwarg)