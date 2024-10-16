"""
This module implements function to solve Inverse Kinematics by numerical methods. 
With providing a set of joint samples from paik solver, we can evaluate the number of iterations of numerical solver compared to random samples.
"""
import time
import numpy as np
from torch import Tensor
from tqdm import tqdm
from common.evaluate import evaluate_pose_error_J3d_P2d
from paik.solver import get_solver
from ikp import get_robot
import pandas as pd
from common.config import Config_IKP



def numerical_IK_iterations(robot, pose: np.ndarray, seeds: np.ndarray):
    """
    Numerical inverse kinematics iterations

    Args:
        pose (np.ndarray): the pose
        seeds (np.ndarray): the seeds

    Returns:
        np.ndarray: the number of iterations
    """
    ik_sols = np.empty((seeds.shape[0], robot.n_dofs))
    num_iters = np.empty(seeds.shape[0])
    
    for i, seed in enumerate(seeds):
        result = robot.inverse_kinematics_klampt(pose, seed=seed, return_iterations=True)
        if result is None:
            ik_sols[i] = np.nan
            num_iters[i] = np.nan
        else:
            ik_sols[i], num_iters[i] = result
    return ik_sols, num_iters

def evaluate_iterations(robot, poses: np.ndarray, seeds: np.ndarray):
    """
    Evaluate the number of iterations of numerical solver compared to random samples

    Args:
        poses (np.ndarray): the poses, shape (num_poses, operation_space_dim)
        seeds (np.ndarray): the seeds, shape (num_sols, num_poses, robot.n_dofs)
    """
    assert seeds.shape[2] == robot.n_dofs, f"Invalid shape of seeds: {seeds.shape}"
    assert seeds.shape[1] == poses.shape[0], f"Invalid shape of seeds: {seeds.shape}"
    
    num_poses = poses.shape[0]
    num_sols = seeds.shape[0]
    
    ik_sols_list = np.zeros((num_sols, num_poses, robot.n_dofs))
    num_iters_list = np.zeros((num_sols, num_poses))
    for i, pose in enumerate(tqdm(poses)):
        ik_sols_list[:, i], num_iters_list[:, i] = numerical_IK_iterations(robot, pose, seeds[:, i])
    
    num_iters_list = num_iters_list.flatten()
    success_rate = np.sum(np.isfinite(num_iters_list)) / num_iters_list.size
    
    
    l2, ang = evaluate_pose_error_J3d_P2d(
        # input J.shape = (num_sols, num_poses, num_dofs or n)
        robot, ik_sols_list, poses, return_all=True
    )
    l2_mm = l2[~np.isnan(l2)].mean() * 1000
    ang_deg = np.rad2deg(ang[~np.isnan(ang)].mean())
    
    # mean number of iterations
    mean_num_iters = num_iters_list[~np.isnan(num_iters_list)]
    return mean_num_iters, success_rate, l2_mm, ang_deg


def main(robot_name: str = "panda"):
    # get solver
    robot = get_robot(robot_name)
    nsf_solver = get_solver(arch_name="nsf", robot_name=robot_name, load=True)
    paik_solver = get_solver(arch_name="paik", robot_name=robot_name, load=True)
    
    num_poses = 50
    num_sols = 250
    
    poses = robot.sample_joint_angles_and_poses(n=num_poses)[1]
    if isinstance(poses, Tensor):
        poses = poses.detach().cpu().numpy()
    
    results = {}
    run_time = {}
    seed_time = {}
    
    begin_time = time.time()
    seeds = robot.sample_joint_angles(n=num_sols*num_poses).reshape(num_sols, num_poses, robot.n_dofs)
    seed_time["random"] = time.time() - begin_time
    results["random"] = evaluate_iterations(robot, poses, seeds)
    run_time["random"] = time.time() - begin_time   
    
    begin_time = time.time()
    seeds = nsf_solver.generate_ik_solutions(poses, num_sols=num_sols).reshape(num_sols, num_poses, robot.n_dofs)
    seed_time["nsf"] = time.time() - begin_time
    results["nsf"] = evaluate_iterations(robot, poses, seeds)
    run_time["nsf"] = time.time() - begin_time
    
    begin_time = time.time()
    seeds = paik_solver.generate_ik_solutions(poses, num_sols=num_sols).reshape(num_sols, num_poses, robot.n_dofs)
    seed_time["paik"] = time.time() - begin_time
    results["paik"] = evaluate_iterations(robot, poses, seeds)
    run_time["paik"] = time.time() - begin_time
    
    print(f"Robot: {robot_name}")
    for key, (mean_num_iters, success_rate, l2_mm, ang_deg) in results.items():
        # print the statistics of the generated solutions
        print(f"{key.upper()}:")
        # only keep the digits before the decimal point
        print(f"Mean number of iterations: {mean_num_iters.mean():.2f}")
        print(f"Success rate: {success_rate*100:.0f}%")
        print(f"Run time: {run_time[key]:.2f} ms")
        print(f"Seed time: {seed_time[key]:.2f} ms")
        print(f"Iteration time: {(run_time[key] - seed_time[key]):.2f} ms")
        print(f"Mean L2 error (mm): {l2_mm:.2f}")
        print(f"Mean angular error (deg): {ang_deg:.2f}")
        print("\n")
        
    # save the results, run_time, and seed_time to csv.
    df = pd.DataFrame({
        "solver": list(results.keys()),
        "mean_num_iters": [results[key][0].mean() for key in results.keys()],
        "success_rate": [results[key][1] for key in results.keys()],
        "run_time": [run_time[key] for key in results.keys()],
        "seed_time": [seed_time[key] for key in results.keys()],
        "iteration_time": [run_time[key] - seed_time[key] for key in results.keys()],
        "mean_L2_error": [results[key][2] for key in results.keys()],
        "mean_angular_error": [results[key][3] for key in results.keys()]
    })
    config = Config_IKP()
    df_file_path = f"{config.record_dir}/iter_time_{robot_name}_{num_poses}_{num_sols}.csv"
    df.to_csv(df_file_path)
    

if __name__ == "__main__":
    robot_names = ["panda", "fetch", "fetch_arm", "atlas_arm", "atlas_waist_arm", "baxter_arm"]
    
    # for robot_name in robot_names:
    main(robot_name=robot_names[-1])