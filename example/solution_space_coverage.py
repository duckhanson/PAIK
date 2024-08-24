# Import required packages
import time
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from paik.solver import get_solver

from common.config import Config_Diversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import (
    mmd_evaluate_multiple_poses,
    make_batches,
    batches_back_to_array,
)


def numerical_inverse_kinematics_single(solver, pose, num_sols):
    seeds, _ = solver.robot.sample_joint_angles_and_poses(n=num_sols)

    ik_sols = np.empty((num_sols, solver.n))
    for i, seed in enumerate(seeds):
        ik_sols[i] = solver.robot.inverse_kinematics_klampt(
            pose=pose, seed=seed
        ) # type: ignore
    return ik_sols

def numerical_inverse_kinematics_batch(solver, P, num_sols) -> np.ndarray[Any, Any]:
    # return shape: (1, num_poses*num_sols, n)
    return np.asarray(
        [numerical_inverse_kinematics_single(solver, p, num_sols) for p in tqdm(P)]
    ).reshape(1, -1, solver.n)
    
def paik_batch(solver: Solver, P, num_sols, std=0.001):
    # P.shape = (num_poses, m)
    # return shape: (1, num_poses*num_sols, n)
    
    # shape: (num_poses, num_sols, m)
    P_num_sols = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
    solver.base_std = std
    
    # shape: (num_poses * num_sols, 1)
    F = solver.get_reference_partition_label(P=P_num_sols[:, 0], num_sols=num_sols)

    # shape: (num_poses * num_sols, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
    
    # shape: (1, num_poses * num_sols, n)
    J_hat = solver.solve_batch(P_num_sols, F, 1)
    
    # return shape: (1, num_poses*num_sols, n)
    return J_hat

def nsf_batch(solver: Solver, P, num_sols, std=0.001):
    # shape: (num_poses, num_sols, m)
    P_num_sols = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
    solver.base_std = std
    
    # shape: (num_poses * num_sols, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
    
    # shape: (num_poses * num_sols)
    F = np.zeros((P_num_sols.shape[0], 1))
    
    # shape: (1, num_poses * num_sols, n)
    J_hat = solver.solve_batch(P_num_sols, F, 1)
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
    # print solve_fn_batch function name
    print(f"Solver: {solve_fn_batch.__name__} starts ...") if verbose else None
    
    # shape: (num_poses, num_sols, num_dofs or n)
    num_poses = P.shape[0]
    begin = time.time()
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
    print(f"Time to solve {num_sols} solutions (avg over {num_poses} poses): {(time.time() - begin)/num_poses *1000:.1f}ms") if verbose else None
    df = pd.DataFrame({"l2 (mm)": l2_mm, "ang (deg)": ang_deg})
    print(df.describe()) if verbose else None
    return J_hat.reshape(num_poses, num_sols, -1), l2_mm[~np.isnan(l2_mm)].mean(), l2_mm[~np.isnan(l2_mm)].std(), ang_deg[~np.isnan(ang_deg)].mean(), ang_deg[~np.isnan(ang_deg)].std()

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
    num_poses = P.shape[0]
    J_hat_array = []
    l2_array = np.empty((len(num_sols_array)))
    l2_std_array = np.empty((len(num_sols_array)))
    ang_array = np.empty((len(num_sols_array)))
    ang_std_array = np.empty((len(num_sols_array)))
    
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        # copy the first solution to the rest of the num_sols
        num_sols_max = np.max(num_sols_array)
        J_hat, l2, l2_std, ang, ang_std = random_ikp(solver, P, num_sols_max, solve_fn_batch)
        J_hat_array = [J_hat[:, num_sols] for num_sols in num_sols_array]
        l2_array = np.full((len(num_sols_array)), l2)
        l2_std_array = np.full((len(num_sols_array)), l2_std)
        ang_array = np.full((len(num_sols_array)), ang)
        ang_std_array = np.full((len(num_sols_array)), ang_std)
    else:
        for i, num_sols in enumerate(num_sols_array):
            J_hat, l2_array[i], l2_std_array[i], ang_array[i], ang_std_array[i] = random_ikp(solver, P, num_sols, solve_fn_batch, std, verbose)
            J_hat_array.append(J_hat)
    return J_hat_array, l2_array, l2_std_array, ang_array, ang_std_array

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
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        num_poses_max = np.max(num_poses_array)
        _, P = solver.robot.sample_joint_angles_and_poses(n=num_poses_max)
        J_hat, l2, l2_std, ang, ang_std = random_ikp(solver, P, num_sols, solve_fn_batch)
        J_hat_array = [J_hat[:num_poses] for num_poses in num_poses_array]
        l2_array = np.full((len(num_poses_array)), l2)
        l2_std_array = np.full((len(num_poses_array)), l2_std)
        ang_array = np.full((len(num_poses_array)), ang)
        ang_std_array = np.full((len(num_poses_array)), ang_std)
    else:
        J_hat_array = []
        l2_array = np.empty((len(num_poses_array)))
        l2_std_array = np.empty((len(num_poses_array)))
        ang_array = np.empty((len(num_poses_array)))
        ang_std_array = np.empty((len(num_poses_array)))
        
        for i, num_poses in enumerate(num_poses_array):
            _, P = solver.robot.sample_joint_angles_and_poses(n=num_poses)
            J_hat, l2_array[i], l2_std_array[i], ang_array[i], ang_std_array[i] = random_ikp(solver, P, num_sols, solve_fn_batch, std, verbose)
            J_hat_array.append(J_hat)
    return J_hat_array, l2_array, l2_std_array, ang_array, ang_std_array

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
    J_hat_array = np.empty((len(std_array), num_poses, num_sols, solver.n))
    l2_array = np.empty((len(std_array)))
    l2_std_array = np.empty((len(std_array)))
    ang_array = np.empty((len(std_array)))
    ang_std_array = np.empty((len(std_array)))
    
    if solve_fn_batch.__name__ == "numerical_inverse_kinematics_batch":
        # copy the first solution to the rest of the stds
        for i, std in enumerate(std_array):
            if i == 0:
                J_hat_array[0], l2_array[0], l2_std_array[0], ang_array[0], ang_std_array[0] = random_ikp(solver, P, num_sols, solve_fn_batch)
            else:
                J_hat_array[i], l2_array[i], l2_std_array[i], ang_array[i], ang_std_array[i] = J_hat_array[0], l2_array[0], l2_std_array[0], ang_array[0], ang_std_array[0]
    else:
        for i, std in enumerate(std_array):
            J_hat_array[i], l2_array[i], l2_std_array[i], ang_array[i], ang_std_array[i] = random_ikp(solver, P, num_sols, solve_fn_batch, std, verbose)
    
    return J_hat_array, l2_array, l2_std_array, ang_array, ang_std_array

def mmd(J_arr1, J_arr2):
    mmd_arr = np.empty((len(J_arr1)))
    for i in range(len(J_arr1)):
        num_poses = len(J_arr1[i])
        if len(J_arr2[i]) != num_poses:
            raise ValueError(f"The number of poses must be the same for both J_arr1[i] ({J_arr1[i].shape}) and J_arr2[i] ({J_arr2[i].shape})")
        mmd_arr[i] = mmd_evaluate_multiple_poses(J_arr1[i], J_arr2[i], num_poses)
    return mmd_arr

def plot_dataframe_mmd_l2_ang(df: pd.DataFrame, record_dir: str, x_axis: str, plot_std: bool=False):
    """
    Plot and save a line chart for the mmd scores, l2 errors, and angular errors with respect to the number of solutions

    Args:
        df (pd.DataFrame): the dataframe with columns "num_sols", "mmd", "l2", "ang"
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
    
    # plot and save a line chart for the mmd scores with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["mmd_num", "mmd_nsf", "mmd_paik"], title=f"MMD Score vs {x_label}", xlabel=x_label, ylabel="MMD Score")    
    fig = ax.get_figure()
    path = f"{record_dir}/mmd_score_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the mmd score line chart to {path}")
    
    # plot and save a line chart for the l2 errors and fill between l2 errors and the range of l2 std with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["l2_num", "l2_nsf", "l2_paik"], title=f"L2 Error (mm) vs {x_label}", xlabel=x_label, ylabel="L2 Error (mm)")
    ax.set_ylim(ymax=15)
    if plot_std:
        ax.fill_between(df[x_axis], df["l2_num"] - df["l2_std_num"], df["l2_num"] + df["l2_std_num"], alpha=0.2)
        ax.fill_between(df[x_axis], df["l2_nsf"] - df["l2_std_nsf"], df["l2_nsf"] + df["l2_std_nsf"], alpha=0.2)
        ax.fill_between(df[x_axis], df["l2_paik"] - df["l2_std_paik"], df["l2_paik"] + df["l2_std_paik"], alpha=0.2)
    fig = ax.get_figure()
    path = f"{config.record_dir}/l2_error_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the l2 error line chart to {path}")
    
    # plot and save a line chart for the angular errors and fill between angular errors and the range of angular std with respect to the number of solutions
    ax = df.plot(x=x_axis, y=["ang_num", "ang_nsf", "ang_paik"], title=f"Angular Error (deg) vs {x_label}", xlabel=x_label, ylabel="Angular Error (deg)")
    ax.set_ylim(ymax=10)
    if plot_std:
        ax.fill_between(df[x_axis], df["ang_num"] - df["ang_std_num"], df["ang_num"] + df["ang_std_num"], alpha=0.2)
        ax.fill_between(df[x_axis], df["ang_nsf"] - df["ang_std_nsf"], df["ang_nsf"] + df["ang_std_nsf"], alpha=0.2)
        ax.fill_between(df[x_axis], df["ang_paik"] - df["ang_std_paik"], df["ang_paik"] + df["ang_std_paik"], alpha=0.2)
    fig = ax.get_figure()
    path = f"{config.record_dir}/ang_error_over_{x_axis}.png"
    fig.savefig(path)
    print(f"save the angular error line chart to {path}")
    
    # save the dataframe to a csv file
    path = f"{config.record_dir}/diversity_over_{x_axis}.csv"
    df.to_csv(path, index=False)
    print(f"save the dataframe to {path}")
    
def plot_iterate_over_num_sols_array(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver):
    num_sols_array = np.array([50, 100, 200, 500, 1000, 2000])

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    
    J_arr_num, l2_arr_num, l2_std_num, ang_arr_num, ang_std_num = iterate_over_num_sols_array(nsf_solver, P, num_sols_array, numerical_inverse_kinematics_batch, verbose=False)
    J_arr_nsf, l2_arr_nsf, l2_std_nsf, ang_arr_nsf, ang_std_nsf = iterate_over_num_sols_array(nsf_solver, P, num_sols_array, nsf_batch, verbose=False)
    J_arr_paik, l2_arr_paik, l2_std_paik, ang_arr_paik, ang_std_paik = iterate_over_num_sols_array(paik_solver, P, num_sols_array, paik_batch, verbose=False)
    
    mmd_num = np.zeros((len(num_sols_array)))
    mmd_nsf = mmd(J_arr_nsf, J_arr_num)
    mmd_paik = mmd(J_arr_paik, J_arr_num)
    
    df = pd.DataFrame({"num_sols": num_sols_array, "mmd_num": mmd_num, "l2_num": l2_arr_num, "l2_std_num": l2_std_num, "ang_num": ang_arr_num, "ang_std_num": ang_std_num, "mmd_nsf": mmd_nsf, "l2_nsf": l2_arr_nsf, "l2_std_nsf": l2_std_nsf, "ang_nsf": ang_arr_nsf, "ang_std_nsf": ang_std_nsf, "mmd_paik": mmd_paik, "l2_paik": l2_arr_paik, "l2_std_paik": l2_std_paik, "ang_paik": ang_arr_paik, "ang_std_paik": ang_std_paik})
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "num_sols")

def plot_iterate_over_num_poses_array(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver):
    num_poses_array = np.array([50, 100, 200, 500, 1000, 2000])
    
    J_arr_num, l2_arr_num, l2_std_num, ang_arr_num, ang_std_num = iterate_over_num_poses_array(nsf_solver, num_poses_array, config.num_sols, numerical_inverse_kinematics_batch, verbose=False)
    J_arr_nsf, l2_arr_nsf, l2_std_nsf, ang_arr_nsf, ang_std_nsf = iterate_over_num_poses_array(nsf_solver, num_poses_array, config.num_sols, nsf_batch, verbose=False)
    J_arr_paik, l2_arr_paik, l2_std_paik, ang_arr_paik, ang_std_paik = iterate_over_num_poses_array(paik_solver, num_poses_array, config.num_sols, paik_batch, verbose=False)
    
    mmd_num = np.zeros((len(num_poses_array)))
    mmd_nsf = mmd(J_arr_nsf, J_arr_num)
    mmd_paik = mmd(J_arr_paik, J_arr_num)
    
    df = pd.DataFrame({"num_poses": num_poses_array, "mmd_num": mmd_num, "l2_num": l2_arr_num, "l2_std_num": l2_std_num, "ang_num": ang_arr_num, "ang_std_num": ang_std_num, "mmd_nsf": mmd_nsf, "l2_nsf": l2_arr_nsf, "l2_std_nsf": l2_std_nsf, "ang_nsf": ang_arr_nsf, "ang_std_nsf": ang_std_nsf, "mmd_paik": mmd_paik, "l2_paik": l2_arr_paik, "l2_std_paik": l2_std_paik, "ang_paik": ang_arr_paik, "ang_std_paik": ang_std_paik})
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "num_poses")

def plot_iterate_over_stds(config: Config_Diversity, nsf_solver: Solver, paik_solver: Solver):
    stds = np.array([0.001, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.6]) 

    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    
    J_arr_num, l2_arr_num, l2_std_num, ang_arr_num, ang_std_num = iterate_over_stds_array(nsf_solver, P, stds, config.num_sols, numerical_inverse_kinematics_batch, verbose=False)
    J_arr_nsf, l2_arr_nsf, l2_std_nsf, ang_arr_nsf, ang_std_nsf = iterate_over_stds_array(nsf_solver, P, stds, config.num_sols, nsf_batch, verbose=False)
    J_arr_paik, l2_arr_paik, l2_std_paik, ang_arr_paik, ang_std_paik = iterate_over_stds_array(paik_solver, P, stds, config.num_sols, paik_batch, verbose=False)
    
    mmd_num = np.zeros((len(stds)))
    mmd_nsf = mmd(J_arr_nsf, J_arr_num)
    mmd_paik = mmd(J_arr_paik, J_arr_num)
    
    df = pd.DataFrame({"stds": stds, "mmd_num": mmd_num, "l2_num": l2_arr_num, "l2_std_num": l2_std_num, "ang_num": ang_arr_num, "ang_std_num": ang_std_num, "mmd_nsf": mmd_nsf, "l2_nsf": l2_arr_nsf, "l2_std_nsf": l2_std_nsf, "ang_nsf": ang_arr_nsf, "ang_std_nsf": ang_std_nsf, "mmd_paik": mmd_paik, "l2_paik": l2_arr_paik, "l2_std_paik": l2_std_paik, "ang_paik": ang_arr_paik, "ang_std_paik": ang_std_paik})
    
    plot_dataframe_mmd_l2_ang(df, config.record_dir, "stds")

if __name__ == "__main__":
    config = Config_Diversity()
    config.num_poses = 500
    config.num_sols = 200

    nsf_solver = get_solver(arch_name="nsf", robot_name="panda", load=True, work_dir=config.workdir)
    paik_solver = get_solver(arch_name="paik", robot_name="panda", load=True, work_dir=config.workdir)

    plot_iterate_over_num_sols_array(config, nsf_solver, paik_solver)    
    plot_iterate_over_stds(config, nsf_solver, paik_solver)
    plot_iterate_over_num_poses_array(config, nsf_solver, paik_solver)
    
    # load df from csv
    # path = f"{config.record_dir}/diversity_over_stds.csv"
    # df = pd.read_csv(path)
    # print(df)
    # plot_dataframe_mmd_l2_ang(df, config.record_dir, "stds")