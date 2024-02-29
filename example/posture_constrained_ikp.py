# Import required packages
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import DEFAULT_NSF, DEFULT_SOLVER

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

NUM_POSES = 300  # 100
NUM_SOLS = 100  # 1000
BATCH_SIZE = 5000
SUCCESS_THRESHOLD = (5e-3, 2)
STD = 0.25
WORKDIR = "/home/luca/paik"
USE_NSF_ONLY = False
METHOD_OF_SELECT_REFERENCE_POSTURE = "knn"


def paik():
    batch_size = BATCH_SIZE
    num_sols = NUM_SOLS
    verbose = True

    solver_param = DEFAULT_NSF if USE_NSF_ONLY else DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver_param.select_reference_posture_method = METHOD_OF_SELECT_REFERENCE_POSTURE
    solver = Solver(solver_param=solver_param)

    J, P = solver.robot.sample_joint_angles_and_poses(n=NUM_POSES)

    # Data Preprocessing
    F = solver.F[solver.J_knn.kneighbors(J, return_distance=False).flatten()]

    # Begin inference
    J_hat = solver.solve_batch(
        P, F, num_sols=num_sols, batch_size=batch_size, verbose=verbose
    )

    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
    # # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # # J.shape = (num_poses, num_dofs or n)
    J_hat = J_hat.reshape(-1, solver.robot.n_dofs)
    J = np.tile(J, (num_sols, 1))
    # # distance_J.shape = (num_sols * num_poses)
    distance_J = np.linalg.norm(J_hat - J, axis=-1).flatten()

    ang = np.rad2deg(ang)
    distance_J = np.rad2deg(distance_J)

    df = pd.DataFrame(
        {
            "l2 (m)": l2,
            "ang (deg)": ang,
            "distance_J (deg)": distance_J,
        }
    )

    print(df.describe())


def ikflow():
    set_seed()

    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    J, P = ik_solver.robot.sample_joint_angles_and_poses(n=NUM_POSES)
    l2 = np.zeros((NUM_SOLS, NUM_POSES))
    ang = np.zeros((NUM_SOLS, NUM_POSES))
    J_hat = torch.empty(
        (NUM_SOLS, NUM_POSES, ik_solver.robot.n_dofs), dtype=torch.float32, device="cpu"
    )
    begin = time.time()
    if NUM_POSES < NUM_SOLS:
        for i in trange(NUM_POSES):
            J_hat[:, i, :] = ik_solver.solve(
                P[i],
                n=NUM_SOLS,
                latent_scale=STD,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2[:, i], ang[:, i] = solution_pose_errors(
                ik_solver.robot, J_hat[:, i, :], P[i]
            )
    else:
        for i in trange(NUM_SOLS):
            J_hat[i] = ik_solver.solve_n_poses(
                P, latent_scale=STD, refine_solutions=False, return_detailed=False
            ).cpu()
            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J_hat[i], P)

    J = np.tile(J, (NUM_SOLS, 1))
    J_hat = J_hat.reshape(-1, ik_solver.robot.n_dofs)
    # # distance_J.shape = (num_sols * num_poses)
    distance_J = np.linalg.norm(J_hat - J, axis=-1)
    ang = np.rad2deg(ang)
    distance_J = np.rad2deg(distance_J)

    l2 = l2.flatten()
    ang = ang.flatten()
    distance_J = distance_J.flatten()

    df = pd.DataFrame(
        {
            "l2 (m)": l2,
            "ang (deg)": ang,
            "distance_J (deg)": distance_J,
        }
    )

    print(df.describe())


if __name__ == "__main__":
    paik()
    ikflow()
