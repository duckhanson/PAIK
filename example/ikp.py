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
    # return shape: (1, num_poses*num_sols, n)
    return np.asarray(
        [numerical_inverse_kinematics_single(solver, p, num_sols) for p in tqdm(P)]
    ).reshape(1, -1, solver.n)
    

def solver_batch(solver, P, num_sols, std=0.001):
    solver.base_std = std

    # shape: (num_poses, num_sols, m)
    P_num_sols = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
    # shape: (num_poses * num_sols, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
    
    if isinstance(solver, PAIK):
        F =  solver.get_reference_partition_label(P=P, num_sols=num_sols)
        # shape: (1, num_poses * num_sols, n)
        J_hat = solver.generate_ik_solutions(P=P_num_sols, F=F)
    else:
        J_hat = solver.generate_ik_solutions(P=P_num_sols)
    
    return J_hat

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
        np.ndarray, float, float: the generated IK solutions with shape (num_poses, num_sols, num_dofs or n).
        float: the mean of l2 error (m).
        float: the mean of angular error (rad).
    """
    if verbose:
        if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
            print(f"Solver: NUM starts ...")
        else:
            print(f"Solver: {solver.__class__.__name__} starts ...") 
        
    # shape: (num_poses, num_sols, num_dofs or n)
    num_poses = P.shape[0]
    begin = time()
    if std is None:
        J_hat = solve_fn_batch(solver, P, num_sols)
    else:
        J_hat = solve_fn_batch(solver, P, num_sols, std)

    l2, ang = solver.evaluate_pose_error_J3d_P2d(
        # input J.shape = (num_sols, num_poses, num_dofs or n)
        J_hat.reshape(num_poses, num_sols, solver.n).transpose(1, 0, 2), P, return_all=True
    )
    l2_mm = l2 * 1000
    ang_deg = np.rad2deg(ang)
    solve_time_ms = (time() - begin)/num_poses *1000
    print(f"Time to solve {num_sols} solutions (avg over {num_poses} poses): {solve_time_ms:.1f}ms") if verbose else None
    df = pd.DataFrame({"l2 (mm)": l2_mm, "ang (deg)": ang_deg})
    print(df.describe()) if verbose else None
    return J_hat.reshape(num_poses, num_sols, -1), l2_mm[~np.isnan(l2_mm)].mean(), ang_deg[~np.isnan(ang_deg)].mean(), solve_time_ms

def mmd(J1, J2):
    """
    Calculate the maximum mean discrepancy between two sets of joint angles

    Args:
        J1 (np.ndarray): the first set of joint angles with shape (num_poses, num_sols, num_dofs or n)
        J2 (np.ndarray): the second set of joint angles with shape (num_poses, num_sols, num_dofs or n)
        
    Returns:
        float: the maximum mean discrepancy between the two sets of joint angles
    """
    num_poses = len(J1)
    if len(J2) != num_poses:
        raise ValueError(f"The number of poses must be the same for both J1 ({J1.shape}) and J2 ({J2[i].shape})")
    mmd_score = mmd_evaluate_multiple_poses(J1, J2, num_poses)
    return mmd_score

if __name__ == "__main__":
    robot_names = ["panda"] # ["panda", "fetch", "fetch_arm", "iiwa7", "atlas_arm", "atlas_waist_arm", "baxter_arm"]
    config = Config_IKP()

    for robot_name in robot_names:
        nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True)
        paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True)
        
        _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
        num_results = random_ikp(nsf_solver, P, config.num_sols, numerical_inverse_kinematics_batch, verbose=False)

        nsf_results = random_ikp(nsf_solver, P, config.num_sols, solver_batch, std=config.std, verbose=False)
        paik_results = random_ikp(paik_solver, P, config.num_sols, solver_batch, std=config.std, verbose=False)
        
        nsf_mmd = mmd(num_results[0], nsf_results[0])
        paik_mmd = mmd(num_results[0], paik_results[0])
        
        print_out_format = "l2: {:1.2f} mm, ang: {:1.2f} deg, time: {:1.1f} ms"

        print(f"Robot: {robot_name} computes {config.num_sols} solutions and average over {config.num_poses} poses")
        # print the results of the random IKP with l2_mm, ang_deg, and solve_time_ms
        print(f"NUM:  {print_out_format.format(*num_results[1:])}")
        print(f"NSF:  {print_out_format.format(*nsf_results[1:])}", f", MMD: {nsf_mmd}")
        print(f"PAIK: {print_out_format.format(*paik_results[1:])}", f", MMD: {paik_mmd}")
        
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
        df_file_path = f"{config.record_dir}/ikp_{robot_name}_{config.num_poses}_{config.num_sols}_{config.std}.csv"
        df.to_csv(df_file_path, index=False)
        print(f"Results are saved to {df_file_path}")
        
    
    
