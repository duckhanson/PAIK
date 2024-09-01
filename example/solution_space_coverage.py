# Import required packages
import time
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from paik.solver import Solver, NSF, PAIK, get_solver

from common.config import Config_Diversity
from common.evaluate import (
    mmd_evaluate_multiple_poses,
)
from ikp import numerical_inverse_kinematics_batch, solver_batch, random_ikp

def iterate_over_num_sols_array(solver, P: np.ndarray, num_sols_array: np.ndarray, solve_fn_batch: Any, std: float=None, verbose: bool=False):
    """
    Iterate over an array of num_sols and generate random IK solutions for each num_sols

    Args:
        solver (Solver): the paik solver object
        P (np.ndarray): the poses with shape (num_poses, m)
        num_sols_array (np.ndarray): the array of num_sols to iterate over
        solve_fn_batch (Any): the batch solver function
        std (float, optional): the standard deviation for the solver. Defaults to None.
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.
        
    Returns:
        np.ndarray, np.ndarray, np.ndarray: the generated IK solutions with shape (num_poses, num_sols, num_dofs or n).
        np.ndarray, np.ndarray: the mean of l2 error (m) and the mean of angular error (rad) for each num_sols
    """
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        # copy the first solution to the rest of the num_sols
        num_sols_max = np.max(num_sols_array)
        J_hat, l2, ang, time_ = random_ikp(solver, P, num_sols_max, solve_fn_batch, verbose)
        J_hat_array = [J_hat[:num_sols] for num_sols in num_sols_array]
        l2_array = np.full((len(num_sols_array)), l2)
        ang_array = np.full((len(num_sols_array)), ang)
        time_array = np.full((len(num_sols_array)), time_)
    else:
        J_hat_array = []
        l2_array = np.empty((len(num_sols_array)))
        ang_array = np.empty((len(num_sols_array)))
        time_array = np.empty((len(num_sols_array)))
        for i, num_sols in enumerate(num_sols_array):
            J_hat, l2_array[i], ang_array[i], time_array[i] = random_ikp(solver, P, num_sols, solve_fn_batch, std, verbose)
            J_hat_array.append(J_hat)
    return J_hat_array, l2_array, ang_array, time_array

def iterate_over_num_poses_array(solver, num_poses_array: np.ndarray, num_sols: int, solve_fn_batch: Any, std: float=None, verbose: bool=False):
    """
    Iterate over an array of num_poses and generate random IK solutions for each num_poses

    Args:
        solver (Solver): the paik solver object
        num_poses_array (np.ndarray): the array of number of poses
        num_sols (int): the number of solutions per pose to generate
        solve_fn_batch (Any): the batch solver function
        std (float, optional): the standard deviation for the solver. Defaults to None.
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.
        
    Returns:
        np.ndarray, np.ndarray, np.ndarray: the generated IK solutions with shape (num_poses, num_sols, num_dofs or n).
        np.ndarray, np.ndarray: the mean of l2 error (m) and the mean of angular error (rad) for each num_poses
    """
    num_poses_max = np.max(num_poses_array)
    _, P = solver.robot.sample_joint_angles_and_poses(n=num_poses_max)
    
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        J_hat, l2, ang, time_ = random_ikp(solver, P, num_sols, solve_fn_batch)
        
        J_hat_array = [J_hat[:, :num_poses] for num_poses in num_poses_array]
        l2_array = np.full((len(num_poses_array)), l2)
        ang_array = np.full((len(num_poses_array)), ang)
        time_array = np.full((len(num_poses_array)), time_)
    else:
        J_hat_array = []
        l2_array = np.empty((len(num_poses_array)))
        ang_array = np.empty((len(num_poses_array)))
        time_array = np.empty((len(num_poses_array)))
        
        for i, num_poses in enumerate(num_poses_array):
            J_hat, l2_array[i], ang_array[i], time_array[i] = random_ikp(solver, P[:num_poses], num_sols, solve_fn_batch, std, verbose)
            J_hat_array.append(J_hat)
    return J_hat_array, l2_array, ang_array, time_array

def iterate_over_stds_array(solver, P: np.ndarray, std_array: np.ndarray, num_sols: int, solve_fn_batch: Any, verbose: bool=False):
    """
    Iterate over an array of stds and generate random IK solutions for each std

    Args:
        solver (Solver): the paik solver object
        P (np.ndarray): the poses with shape (num_poses, m)
        std_array (np.ndarray): the array of stds to iterate over
        num_sols (int): the number of solutions per pose to generate
        solve_fn_batch (Any): the batch solver function
        verbose (bool, optional): print the statistics of the generated solutions. Defaults to False.
        
    Returns:
        np.ndarray, np.ndarray, np.ndarray: the generated IK solutions with shape (num_poses, num_sols, num_dofs or n).
        np.ndarray, np.ndarray: the mean of l2 error (m) and the mean of angular error (rad) for each std
    """
    num_poses = P.shape[0]
    
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        # copy the first solution to the rest of the stds
        J_hat, l2, ang, time_ = random_ikp(solver, P, num_sols, solve_fn_batch)
        
        # shape: (num_poses, num_sols, num_dofs or n) for each std
        # copy the first solution to the rest of the stds using np.repeat and np.expand_dims
        J_hat_array = np.expand_dims(J_hat, axis=0).repeat(len(std_array), axis=0)
        l2_array = np.full((len(std_array)), l2)
        ang_array = np.full((len(std_array)), ang)
        time_array = np.full((len(std_array)), time_)
    else:
        J_hat_array = np.empty((len(std_array), num_sols, num_poses, solver.n))
        l2_array = np.empty((len(std_array)))
        ang_array = np.empty((len(std_array)))
        time_array = np.empty((len(std_array)))
        for i, std in enumerate(std_array):
            J_hat_array[i], l2_array[i],ang_array[i], time_array[i] = random_ikp(solver, P, num_sols, solve_fn_batch, std, verbose)
    
    return J_hat_array, l2_array, ang_array, time_array

def mmd_arr(J_arr1, J_arr2):
    arr_size = len(J_arr1)
    # the last second dimension is the number of poses
    mmd_arr_ = np.array([mmd_evaluate_multiple_poses(J_arr1[i], J_arr2[i], J_arr1[i].shape[-2]) for i in range(arr_size)])
    return mmd_arr_

def plot_dataframe_mmd_l2_ang(df: pd.DataFrame, record_dir: str, x_axis: str, plot_std: bool=False):
    """
    Plot and save a line chart for the mmd_arr scores, l2 errors, and angular errors with respect to the number of solutions

    Args:
        df (pd.DataFrame): the dataframe with columns "num_sols", "mmd_arr", "l2", "ang"
        record_dir (str): the directory to save the line chart
        x_axis (str): the x_axis of the line chart
    """
    if x_axis == "num_sols":
        x_label = "Number of Solutions"
    elif x_axis == "stds":
        x_label = "Standard Deviation of Random Seeds"
    elif x_axis == "num_poses":
        x_label = "Number of Poses"
    else:
        raise ValueError(f"x_axis must be either 'num_sols' or 'stds'")
    
    # plot and save a line chart for the mmd_arr scores with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["mmd_nsf", "mmd_paik"], title=f"MMD Score vs {x_label}", xlabel=x_label, ylabel="MMD Score")    
    fig = ax.get_figure()
    path = f"{record_dir}/mmd_score_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the mmd_arr score line chart to {path}")
    
    # plot and save a line chart for the l2 errors with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["l2_num", "l2_nsf", "l2_paik"], title=f"L2 Error (mm) vs {x_label}", xlabel=x_label, ylabel="L2 Error (mm)")
    ax.set_ylim(ymax=15)
    fig = ax.get_figure()
    path = f"{config.record_dir}/l2_error_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the l2 error line chart to {path}")
    
    # plot and save a line chart for the angular errors with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["ang_num", "ang_nsf", "ang_paik"], title=f"Angular Error (deg) vs {x_label}", xlabel=x_label, ylabel="Angular Error (deg)")
    ax.set_ylim(ymax=10)
    fig = ax.get_figure()
    path = f"{config.record_dir}/ang_error_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the angular error line chart to {path}")
    
    # plot and save a line chart for the time with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["time_num", "time_nsf", "time_paik"], title=f"Solve Time (ms) vs {x_label}", xlabel=x_label, ylabel="Solve Time (ms)")
    ax.set_ylim(ymax=80)
    fig = ax.get_figure()
    path = f"{config.record_dir}/solve_time_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the solve time line chart to {path}")
    
    # save the dataframe to a csv file
    path = f"{config.record_dir}/diversity_over_{x_axis}.csv"
    df.to_csv(path, index=False)
    print(f"save the dataframe to {path}")
    
def plot_iterate_over_num_sols_array(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver, verbose: bool=False):
    num_sols_array = np.array([50, 100, 200, 500, 1000, 2000])

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    
    num_results = iterate_over_num_sols_array(nsf_solver, P, num_sols_array, numerical_inverse_kinematics_batch, verbose=verbose)
    nsf_results = iterate_over_num_sols_array(nsf_solver, P, num_sols_array, solver_batch, std=config.std, verbose=verbose)
    paiK_results = iterate_over_num_sols_array(paik_solver, P, num_sols_array, solver_batch, std=config.std, verbose=verbose)
    
    # compute the mmd_arr scores for the NSF solutions, and PAIK solutions
    mmd_nsf = mmd_arr(nsf_results[0], num_results[0])
    mmd_paik = mmd_arr(paiK_results[0], num_results[0])
    
    # each results is a tuple of J_hat, l2, ang, time
    # create a dataframe with results of num, nsf, paik and mmd_arr scores for nsf and paik
    df = pd.DataFrame({
        "num_sols": num_sols_array,
        "mmd_nsf": mmd_nsf,
        "mmd_paik": mmd_paik,
        "l2_num": num_results[1],
        "l2_nsf": nsf_results[1],
        "l2_paik": paiK_results[1],
        "ang_num": num_results[2],
        "ang_nsf": nsf_results[2],
        "ang_paik": paiK_results[2],
        "time_num": num_results[3],
        "time_nsf": nsf_results[3],
        "time_paik": paiK_results[3]
    })
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "num_sols")

def plot_iterate_over_num_poses_array(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver, verbose: bool=False):
    num_poses_array = np.array([50, 100, 200, 500, 1000, 2000])
    
    num_results = iterate_over_num_poses_array(nsf_solver, num_poses_array, config.num_sols, numerical_inverse_kinematics_batch, verbose=verbose)
    nsf_results = iterate_over_num_poses_array(nsf_solver, num_poses_array, config.num_sols, solver_batch, std=config.std, verbose=verbose)
    paik_results = iterate_over_num_poses_array(paik_solver, num_poses_array, config.num_sols, solver_batch, std=config.std, verbose=verbose)
    
    # compute the mmd_arr scores for the NSF solutions, and PAIK solutions
    mmd_nsf = mmd_arr(nsf_results[0], num_results[0])
    mmd_paik = mmd_arr(paik_results[0], num_results[0])
    
    # each results is a tuple of J_hat, l2, ang, time
    # create a dataframe with results of num, nsf, paik and mmd_arr scores for nsf and paik
    df = pd.DataFrame({
        "num_poses": num_poses_array,
        "mmd_nsf": mmd_nsf,
        "mmd_paik": mmd_paik,
        "l2_num": num_results[1],
        "l2_nsf": nsf_results[1],
        "l2_paik": paik_results[1],
        "ang_num": num_results[2],
        "ang_nsf": nsf_results[2],
        "ang_paik": paik_results[2],
        "time_num": num_results[3],
        "time_nsf": nsf_results[3],
        "time_paik": paik_results[3]
    })
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "num_poses")

def plot_iterate_over_stds(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver, verbose: bool=False):
    stds = np.array([0.001, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.6]) 

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    
    num_results = iterate_over_stds_array(nsf_solver, P, stds, config.num_sols, numerical_inverse_kinematics_batch, verbose=verbose)
    nsf_results = iterate_over_stds_array(nsf_solver, P, stds, config.num_sols, solver_batch, verbose=verbose)
    paik_results = iterate_over_stds_array(paik_solver, P, stds, config.num_sols, solver_batch, verbose=verbose)
    
    # compute the mmd_arr scores for the NSF solutions, and PAIK solutions
    mmd_nsf = mmd_arr(nsf_results[0], num_results[0])
    mmd_paik = mmd_arr(paik_results[0], num_results[0])
    
    # each results is a tuple of J_hat, l2, ang, time
    # create a dataframe with results of num, nsf, paik and mmd_arr scores for nsf and paik
    df = pd.DataFrame({
        "stds": stds,
        "mmd_nsf": mmd_nsf,
        "mmd_paik": mmd_paik,
        "l2_num": num_results[1],
        "l2_nsf": nsf_results[1],
        "l2_paik": paik_results[1],
        "ang_num": num_results[2],
        "ang_nsf": nsf_results[2],
        "ang_paik": paik_results[2],
        "time_num": num_results[3],
        "time_nsf": nsf_results[3],
        "time_paik": paik_results[3]
    })    
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "stds")

if __name__ == "__main__":
    config = Config_Diversity()
    
    robot_name = "panda"
    nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True, work_dir=config.workdir)
    paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True, work_dir=config.workdir)

    plot_iterate_over_num_sols_array(config, nsf_solver, paik_solver, verbose=False)    
    plot_iterate_over_stds(config, nsf_solver, paik_solver, verbose=False)
    plot_iterate_over_num_poses_array(config, nsf_solver, paik_solver, verbose=False)
    
    # load df from csv
    # path = f"{config.record_dir}/diversity_over_stds.csv"
    # df = pd.read_csv(path)
    # print(df)
    # plot_dataframe_mmd_l2_ang(df, config.record_dir, "stds")