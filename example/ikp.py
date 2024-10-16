from typing import Any
import numpy as np
from time import time

import pandas as pd
import scipy
from tqdm import tqdm

from common.evaluate import evaluate_pose_error_J3d_P2d, mmd_J3d_J3d
from paik.solver import NSF, PAIK, Solver, get_solver
from sklearn.cluster import DBSCAN
from common.config import Config_IKP
from common.display import save_ikp
from scipy.spatial.distance import cdist


import os

from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
import jrl.robots as jrlib
import paik.klampt_robot as chlib

import torch
from ikflow.training.lt_model import IkfLitModel
from experiments_ikflow import get_ikflow_solver

# set the same random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

DEFAULT_WEIGHTS_PATH = {
    'atlas_arm': '/home/luca/ikflow/training_logs/atlas_arm--Sep.11.2024_07:52:PM/ikflow-checkpoint-step=90000.ckpt',
    'atlas_waist_arm': '/home/luca/ikflow/training_logs/atlas_waist_arm--Sep.10.2024_11:3:PM/ikflow-checkpoint-step=270000.ckpt',
    'baxter_arm': '/home/luca/ikflow/training_logs/baxter--Sep.12.2024_09:27:AM/ikflow-checkpoint-step=960000.ckpt',
}

def get_robot(robot_name: str):
    try:
        robot = jrlib.get_robot(robot_name)
    except ValueError:
        try:
            robot = chlib.get_robot(robot_name)
        except ValueError:
            raise ValueError(
                f"Error: Robot '{robot_name}' not found in either jrlib or chlib")

    return robot

def _load_ikflow_local_pretrain(robot_name: str):
    

    ckpt_filepath = DEFAULT_WEIGHTS_PATH[robot_name]

    assert os.path.isfile(
        ckpt_filepath
    ), f"File '{ckpt_filepath}' was not found. Unable to load model weights"

    robot = get_robot(robot_name)

    # Build IKFlowSolver and set weights
    checkpoint = torch.load(
        ckpt_filepath, map_location=lambda storage, loc: storage)
    hyper_parameters: IkflowModelParameters = checkpoint["hyper_parameters"]["base_hparams"]
    ik_solver = IKFlowSolver(hyper_parameters, robot)
    return ik_solver, hyper_parameters, ckpt_filepath


def load_ikflow_solver_from_lightning_checkpoint(robot_name: str):
    if robot_name in DEFAULT_WEIGHTS_PATH:
        ik_solver, hyper_parameters, ckpt_filepath = _load_ikflow_local_pretrain(
            robot_name)
        model = IkfLitModel.load_from_checkpoint(
            ckpt_filepath, ik_solver=ik_solver)
        model.eval()
        ik_solver = model.ik_solver
    else:
        ik_solver, _ = get_ikflow_solver(robot_name)
    return ik_solver


def numerical_inverse_kinematics_single(solver, pose, num_sols):
    ik_sols = np.empty((num_sols, solver.robot.n_dofs))
    for i in range(num_sols):
        ik_sols[i] = solver.robot.inverse_kinematics_klampt(pose)  # type: ignore
    return ik_sols


def numerical_inverse_kinematics_batch(solver, P, num_sols):
    ik_sols_batch = np.empty((num_sols, P.shape[0], solver.robot.n_dofs))
    for i, pose in enumerate(P):
        ik_sols_batch[:, i] = numerical_inverse_kinematics_single(
            solver, pose, num_sols)
    # return shape: (num_sols, num_poses, n)
    return ik_sols_batch


def solver_batch(solver, P, num_sols, std=0.001):
    solver.base_std = std

    # shape: (num_sols, num_poses, m)
    P_num_sols = np.expand_dims(P, axis=0).repeat(num_sols, axis=0)
    # shape: (num_sols*num_poses, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])

    if isinstance(solver, PAIK):
        F = solver.get_reference_partition_label(P=P, num_sols=num_sols)
        # shape: (1, num_sols*num_poses, n)
        J_hat = solver.generate_ik_solutions(P=P_num_sols, F=F, verbose=False)
    elif isinstance(solver, NSF):
        J_hat = solver.generate_ik_solutions(P=P_num_sols, verbose=False)
    else:
        J_hat = np.empty((num_sols, P.shape[0], solver.robot.n_dofs))
        P_torch = torch.tensor(P, dtype=torch.float32).to('cuda')
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
    return J_hat.reshape(num_sols, P.shape[0], -1)


def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian (RBF) kernel between two vectors."""
    return np.exp(-cdist(x, y, 'sqeuclidean') / (2 * sigma ** 2))

def inverse_multiquadric_kernel(x, y, c=1.0):
    """Compute the Inverse Multi-Quadric (IMQ) kernel between two vectors."""
    return 1.0 / np.sqrt(cdist(x, y, 'sqeuclidean') + c**2)

def compute_mmd(X, Y, kernel=gaussian_kernel):
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions."""
    
    # filter out the nan values
    X = X[~np.isnan(X).any(axis=1)]
    Y = Y[~np.isnan(Y).any(axis=1)]
    
    XX = kernel(X, X)
    YY = kernel(Y, Y)
    XY = kernel(X, Y)
    
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def get_number_of_distinct_solutions(J_hat: np.ndarray, eps: float, min_samples: int):
    """
    Get the number of distinct solutions for each pose

    Args:
        J_hat (np.ndarray): the generated IK solutions with shape (num_sols, num_poses, num_dofs or n)

    Returns:
        np.ndarray: the number of distinct solutions for each pose
    """
    
    # if the input is 3D, reshape it to 2D
    if len(J_hat.shape) == 3:
        J_hat = J_hat.reshape(J_hat.shape[0], -1)
    
    # fileter out the nan values
    J_hat = J_hat[~np.isnan(J_hat).any(axis=1)]
        
    # Initialize DBSCAN with chosen parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model
    dbscan.fit(J_hat)

    # Get the labels (clusters)
    labels = dbscan.labels_

    # Count the number of clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # print(f"Number of representatives (clusters): {num_clusters}")

    return num_clusters, labels


def random_ikp(solver: Solver, P: np.ndarray, num_sols: int, solve_fn_batch: Any, std: float = None, verbose: bool = False, return_dict: bool = False):
    """
    Generate random IK solutions for a given solve_fn_batch and poses

    Args:
        solver (Solver): the paik solver object
        P (np.ndarray): the poses with shape (num_poses, m)
        num_sols (int): the number of solutions per pose to generate
        solve_fn_batch (Any): the batch solver function
        std (float, optional): the standard deviation for the solver. Defaults to None.
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.

    Returns:
        np.ndarray, float, float: the generated IK solutions with shape (num_sols, num_poses, num_dofs or n).
        float: the mean of l2 error (mm).
        float: the mean of angular error (deg).
        float: the average time to solve num_sols for a pose (ms).
    """
    if verbose:
        if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
            print(f"Solver: NUM starts ...")
        else:
            print(f"Solver: {solver.__class__.__name__} starts ...")

    # shape: (num_poses, num_sols, num_dofs or n)
    num_poses = P.shape[0]
    begin = time()
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        J_hat = solve_fn_batch(solver, P, num_sols)
    else:
        J_hat = solve_fn_batch(solver, P, num_sols, std)
    assert J_hat.shape == (
        num_sols, num_poses, solver.robot.n_dofs), f"J_hat shape {J_hat.shape} is not correct"

    l2, ang = evaluate_pose_error_J3d_P2d(
        # input J.shape = (num_sols, num_poses, num_dofs or n)
        solver.robot, J_hat, P, return_all=True
    )
    l2_mm = l2[~np.isnan(l2)].mean() * 1000
    ang_deg = np.rad2deg(ang[~np.isnan(ang)].mean())
    solve_time_ms = round((time() - begin) / len(P), 3) * 1000
    num_rep, _ = get_number_of_distinct_solutions(J_hat, eps=0.01, min_samples=5)
    
    if return_dict:
        return {
            "J_hat": J_hat,
            "l2_mm": l2_mm,
            "ang_deg": ang_deg,
            "solve_time_ms": solve_time_ms,
            "#rep": num_rep
        }
    
    return list((J_hat, l2_mm, ang_deg, solve_time_ms, num_rep))

def _test_1_random_pose_ikp_w_mmd(solvers, pose, num_sols, std, verbose=False):
    assert len(pose.shape) == 1, f"Pose shape {pose.shape} is not correct"
    results = {
        key: random_ikp(solver, pose.reshape(1, -1), num_sols, solver_batch, std=std, verbose=verbose, return_dict=True)
        for key, solver in solvers.items()
    }
    results["num"] = random_ikp(solvers["nsf"], pose.reshape(1, -1), num_sols, numerical_inverse_kinematics_batch, verbose=verbose, return_dict=True)
    
    num_ik_solutions = results["num"]["J_hat"].reshape(-1, results["num"]["J_hat"].shape[-1])
    for key in results.keys():
        ik_solutions = results[key]["J_hat"].reshape(-1, results[key]["J_hat"].shape[-1])
        results[key]["mmd_guassian"] = compute_mmd(ik_solutions, num_ik_solutions)
        results[key]["mmd_imq"] = compute_mmd(ik_solutions, num_ik_solutions, kernel=inverse_multiquadric_kernel)
        
    return results


def test_random_ikp_with_mmd(robot_name: str, num_poses: int, num_sols: int, stds: list, record_dir: str, verbose: bool = False):
    """
    Test the random IKP with MMD evaluation

    Args:
        robot_name (str): robot name
        num_poses (int): number of poses  
        num_sols (int): number of solutions per pose
        std (float): standard deviation for the solver
        record_dir (str): the directory to save the results
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.
    """

    solvers = {
        "ikflow": load_ikflow_solver_from_lightning_checkpoint(robot_name=robot_name),
        "nsf": get_solver(arch_name="nsf", robot_name=robot_name, load=True),
        "paik": get_solver(arch_name="paik", robot_name=robot_name, load=True),
    }

    _, P = solvers["nsf"].robot.sample_joint_angles_and_poses(n=num_poses)

    # dummy run to load the model
    # random_ikp(solvers["nsf"], P, num_sols,
    #            solver_batch, std=stds[0], verbose=False)
    
    solver_names = list(solvers.keys()) + ["num"]
    
    for std in stds:
        df_dicts = {}

        for pose in tqdm(P):
            results = _test_1_random_pose_ikp_w_mmd(solvers, pose, num_sols, std, verbose=verbose)
            
            if df_dicts == {}:  # initialize the dictionary
                col_names = list(results["num"].keys())
                for key in solver_names:
                    df_dicts[key] = {col: [] for col in col_names}
            
            for key in solver_names:
                for col in col_names:
                    df_dicts[key][col].append(results[key][col])

        for key, df_dict in df_dicts.items():
            df = pd.DataFrame(df_dict)
            print(f"Solver: {key}, std: {std}, num_poses: {num_poses}, num_sols: {num_sols}")
            print(df.describe())
            df_file_path = f"{record_dir}/ikp_{robot_name}_{num_poses}_{num_sols}_{std}_{key}.csv"
            df.to_csv(df_file_path, index=False)
            print(f"Results are saved to {df_file_path}")


if __name__ == "__main__":
    robot_names = ["panda"] # , "fetch", "fetch_arm", "atlas_arm", "atlas_waist_arm", "baxter_arm"
    config = Config_IKP()
    
    stds = [0.01, 0.1, 0.25]
    for robot_name in robot_names:
        test_random_ikp_with_mmd(robot_name, config.num_poses,
                                config.num_sols, stds, config.record_dir, verbose=False)
