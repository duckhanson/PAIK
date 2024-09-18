from typing import Any
import numpy as np
from time import time

import pandas as pd
from tqdm import tqdm

from common.evaluate import mmd_evaluate_multiple_poses, evaluate_pose_error_J3d_P2d, mmd_J3d_J3d
from paik.solver import NSF, PAIK, Solver, get_solver
from common.config import Config_IKP
from common.display import save_ikp


import os

from ikflow.model import IkflowModelParameters
from ikflow.ikflow_solver import IKFlowSolver
import jrl.robots as jrlib
import paik.klampt_robot as chlib

import torch
from ikflow.training.lt_model import IkfLitModel

# set the same random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

DEFAULT_WEIGHTS_PATH = {
    'atlas_arm': '/home/luca/ikflow/training_logs/atlas_arm--Sep.11.2024_07:52:PM/ikflow-checkpoint-step=90000.ckpt',
    'atlas_waist_arm': '/home/luca/ikflow/training_logs/atlas_waist_arm--Sep.10.2024_11:3:PM/ikflow-checkpoint-step=270000.ckpt',
    'baxter_arm': '/home/luca/ikflow/training_logs/baxter--Sep.12.2024_09:27:AM/ikflow-checkpoint-step=960000.ckpt',
}


def _load_ikflow_local_pretrain(robot_name: str):
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
    ik_solver, hyper_parameters, ckpt_filepath = _load_ikflow_local_pretrain(
        robot_name)
    model = IkfLitModel.load_from_checkpoint(
        ckpt_filepath, ik_solver=ik_solver)
    model.eval()

    return model.ik_solver


def numerical_inverse_kinematics_single(solver, pose, num_sols):
    ik_sols = np.empty((num_sols, solver.robot.n_dofs))
    for i in range(num_sols):
        ik_sols[i] = solver.robot.inverse_kinematics_klampt(
            p=pose)  # type: ignore
    return ik_sols


def numerical_inverse_kinematics_batch(solver, P, num_sols):
    ik_sols_batch = np.empty((num_sols, P.shape[0], solver.robot.n_dofs))
    for i, pose in enumerate(tqdm(P)):
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
        J_hat = solver.generate_ik_solutions(P=P_num_sols, F=F)
    elif isinstance(solver, NSF):
        J_hat = solver.generate_ik_solutions(P=P_num_sols)
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


def random_ikp(solver: Solver, P: np.ndarray, num_sols: int, solve_fn_batch: Any, std: float = None, verbose: bool = False):
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
    return J_hat, l2_mm, ang_deg, solve_time_ms


def test_random_ikp_with_mmd(robot_name: str, num_poses: int, num_sols: int, std: float, record_dir: str, verbose: bool = False):
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
    random_ikp(solvers["nsf"], P, num_sols,
               solver_batch, std=std, verbose=False)

    results = {
        "num": random_ikp(solvers["nsf"], P, num_sols, numerical_inverse_kinematics_batch, verbose=False),
        **{key: random_ikp(solver, P, num_sols, solver_batch, std=std, verbose=True) for key, solver in solvers.items()}
    }

    mmd_results = {
        "num": 0,
        **{key: mmd_J3d_J3d(results[key][0], results["num"][0], num_poses) for key in results.keys() if key != "num"}
    }

    if verbose:
        print_out_format = "l2: {:1.2f} mm, ang: {:1.2f} deg, time: {:1.1f} ms"
        print("="*70)
        print(
            f"Robot: {robot_name} computes {num_sols} solutions and average over {num_poses} poses")
        # print the results of the random IKP with l2_mm, ang_deg, and solve_time_ms
        for key in results.keys():
            print(
                f"{key.upper()}: MMD: {mmd_results[key]}, {print_out_format.format(*results[key][1:])}")
        print("="*70)

    # get count results keys
    num_solvers = len(results.keys())

    df = pd.DataFrame({
        "robot": [robot_name] * num_solvers,
        "solver": list(results.keys()),
        "l2_mm": [results[key][1] for key in results.keys()],
        "ang_deg": [results[key][2] for key in results.keys()],
        "solve_time_ms": [results[key][3] for key in results.keys()],
        "mmd": [mmd_results[key] for key in mmd_results.keys()],
    })

    # save the results to a csv file
    df_file_path = f"{record_dir}/ikp_{robot_name}_{num_poses}_{num_sols}_{std}.csv"
    df.to_csv(df_file_path, index=False)
    print(f"Results are saved to {df_file_path}")


if __name__ == "__main__":
    # ["panda", "fetch", "fetch_arm", "atlas_arm", "atlas_waist_arm", "baxter"]
    robot_names = ["baxter_arm"]
    config = Config_IKP()

    for robot_name in robot_names:
        test_random_ikp_with_mmd(robot_name, config.num_poses,
                                 config.num_sols, config.std, config.record_dir, verbose=True)
