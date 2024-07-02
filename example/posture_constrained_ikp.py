# Import required packages
import numpy as np
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import PANDA_NSF, PANDA_PAIK

from common.config import Config_Posture
from common.display import display_posture
from common.evaluate import compute_distance_J


def paik(config: Config_Posture):
    solver_param = PANDA_PAIK
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)

    J, P = solver.robot.sample_joint_angles_and_poses(n=config.num_poses)

    # Data Preprocessing
    F = solver.F[solver.J_knn.kneighbors(J, return_distance=False).flatten()]

    # Begin inference
    J_hat = solver.solve_batch(
        P, F, num_sols=config.num_sols, batch_size=config.batch_size, verbose=True
    )

    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
    # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # J.shape = (num_poses, num_dofs or n)
    distance_J = compute_distance_J(J_hat, J)
    display_posture(
        config.record_dir,
        "paik",
        l2,
        ang,
        distance_J,
        config.success_distance_thresholds,
    )


def nsf(config: Config_Posture):
    solver_param = PANDA_NSF
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)

    J, P = solver.robot.sample_joint_angles_and_poses(n=config.num_poses)

    # Data Preprocessing
    F = solver.F[solver.J_knn.kneighbors(J, return_distance=False).flatten()]

    # Begin inference
    J_hat = solver.solve_batch(
        P, F, num_sols=config.num_sols, batch_size=config.batch_size, verbose=True
    )

    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
    # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # J.shape = (num_poses, num_dofs or n)
    distance_J = compute_distance_J(J_hat, J)
    display_posture(
        config.record_dir,
        "nsf",
        l2,
        ang,
        distance_J,
        config.success_distance_thresholds,
    )


if __name__ == "__main__":
    config = Config_Posture()
    config.date = "2024_03_02"
    # paik(config)
    nsf(config)
