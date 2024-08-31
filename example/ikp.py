from typing import Any
import numpy as np
from time import time

import pandas as pd
from tqdm import tqdm

from common.evaluate import mmd_evaluate_multiple_poses
from paik.solver import PAIK, Solver, get_solver
from common.config import Config_IKP
from common.display import save_ikp


def numerical_inverse_kinematics_single(solver, pose, num_sols):
    seeds, _ = solver.robot.sample_joint_angles_and_poses(n=num_sols)

    ik_sols = np.empty((num_sols, solver.n))
    for i, seed in enumerate(seeds):
        ik_sols[i] = solver.robot.inverse_kinematics_klampt(
            pose=pose, seed=seed
        ) # type: ignore
    return ik_sols

def numerical_inverse_kinematics_batch(solver, P, num_sols):
    ik_sols_batch = np.empty((num_sols, P.shape[0], solver.n))
    for i, pose in enumerate(tqdm(P)):
        ik_sols_batch[:, i] = numerical_inverse_kinematics_single(solver, pose, num_sols)
    # return shape: (num_sols, num_poses, n)
    return ik_sols_batch
    

def solver_batch(solver, P, num_sols, std=0.001):
    solver.base_std = std

    # shape: (num_sols, num_poses, m)
    P_num_sols = np.expand_dims(P, axis=0).repeat(num_sols, axis=0)
    # shape: (num_sols*num_poses, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
    
    if isinstance(solver, PAIK):
        F =  solver.get_reference_partition_label(P=P, num_sols=num_sols)
        # shape: (1, num_sols*num_poses, n)
        J_hat = solver.generate_ik_solutions(P=P_num_sols, F=F)
    else:
        J_hat = solver.generate_ik_solutions(P=P_num_sols)
    # return shape: (num_sols, num_poses, n)
    return J_hat.reshape(num_sols, P.shape[0], -1)

def random_ikp(solver: Solver, P: np.ndarray, num_sols: int, solve_fn_batch: Any, std: float=None, verbose: bool=False):
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
    assert J_hat.shape == (num_sols, num_poses, solver.n), f"J_hat shape {J_hat.shape} is not correct"

    l2, ang = solver.evaluate_pose_error_J3d_P2d(
        # input J.shape = (num_sols, num_poses, num_dofs or n)
        J_hat, P, return_all=True
    )
    l2_mm = l2 * 1000
    ang_deg = np.rad2deg(ang)
    solve_time_ms = (time() - begin)/num_poses *1000
    return J_hat, l2_mm[~np.isnan(l2_mm)].mean(), ang_deg[~np.isnan(ang_deg)].mean(), solve_time_ms

def mmd(J1, J2, num_poses):
    """
    Calculate the maximum mean discrepancy between two sets of joint angles

    Args:
        J1 (np.ndarray): the first set of joint angles with shape (num_sols, num_poses, num_dofs or n)
        J2 (np.ndarray): the second set of joint angles with shape (num_sols, num_poses, num_dofs or n)
        
    Returns:
        float: the maximum mean discrepancy between the two sets of joint angles
    """
    # check num_poses is the same for J1 and J2
    if J1.shape[1] != num_poses or J2.shape[1] != num_poses:
        raise ValueError(f"The number of poses must be the same for both J1 ({J1.shape[1]}) and J2 ({J1.shape[1]})")
    mmd_score = mmd_evaluate_multiple_poses(J1, J2, num_poses)
    return mmd_score

def random_ikp_with_mmd(robot_name: str, num_poses: int, num_sols: int, std: float, record_dir: str, verbose: bool=False):
    """
    Generate random IK solutions for a given robot and poses

    Args:
        robot_name (str): the name of the robot
        num_poses (int): the number of poses to generate
        num_sols (int): the number of solutions per pose to generate
        std (float): the standard deviation for the solver
        record_dir (str): the directory to save the results
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.
        
    Returns:
        None
    """
    nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True)
    paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True)
    
    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=num_poses)
    num_results = random_ikp(nsf_solver, P, num_sols, numerical_inverse_kinematics_batch, verbose=verbose)

    # dummy run to load the model
    random_ikp(nsf_solver, P, num_sols, solver_batch, std=std, verbose=False)
    nsf_results = random_ikp(nsf_solver, P, num_sols, solver_batch, std=std, verbose=verbose)
    paik_results = random_ikp(paik_solver, P, num_sols, solver_batch, std=std, verbose=verbose)
    
    nsf_mmd = mmd(num_results[0], nsf_results[0], num_poses)
    paik_mmd = mmd(num_results[0], paik_results[0], num_poses)
    

    if verbose:
        print_out_format = "l2: {:1.2f} mm, ang: {:1.2f} deg, time: {:1.1f} ms"
        print("="*70)
        print(f"Robot: {robot_name} computes {num_sols} solutions and average over {num_poses} poses")
        # print the results of the random IKP with l2_mm, ang_deg, and solve_time_ms
        print(f"NUM:  {print_out_format.format(*num_results[1:])}")
        print(f"NSF:  {print_out_format.format(*nsf_results[1:])}", f", MMD: {nsf_mmd}")
        print(f"PAIK: {print_out_format.format(*paik_results[1:])}", f", MMD: {paik_mmd}")
        print("="*70)
    
    # use a dataframe to save the results without J_hat
    # each row is a robot, a solver, and the results of l2_mm, ang_deg, and solve_time_ms, and MMD
    df = pd.DataFrame({
        "robot": [robot_name, robot_name, robot_name],
        "solver": ["NUM", "NSF", "PAIK"],
        "l2_mm": [num_results[1], nsf_results[1], paik_results[1]],
        "ang_deg": [num_results[2], nsf_results[2], paik_results[2]],
        "solve_time_ms": [num_results[3], nsf_results[3], paik_results[3]],
        "MMD": [0, nsf_mmd, paik_mmd]
    })
    
    # save the results to a csv file
    df_file_path = f"{record_dir}/ikp_{robot_name}_{num_poses}_{num_sols}_{std}.csv"
    df.to_csv(df_file_path, index=False)
    print(f"Results are saved to {df_file_path}")

if __name__ == "__main__":
    robot_names = ["atlas_waist_arm"] # ["panda", "fetch", "fetch_arm", "atlas_arm", "atlas_waist_arm", "baxter_arm"]
    config = Config_IKP()

    for robot_name in robot_names:
        random_ikp_with_mmd(robot_name, config.num_poses, config.num_sols, config.std, config.record_dir, verbose=True)
        
    
    
