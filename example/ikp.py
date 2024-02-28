# Import required packages
import time
import numpy as np
from tabulate import tabulate
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import DEFAULT_NSF, DEFULT_SOLVER

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

NUM_POSES = 3000  # 100
NUM_SOLS = 1000  # 1000
BATCH_SIZE = 5000
SUCCESS_THRESHOLD = (5e-3, 2)
STD = 0.25
WORKDIR = "/home/luca/paik"
USE_NSF_ONLY = False
METHOD_OF_SELECT_REFERENCE_POSTURE = "knn"


def paik():
    solver_param = DEFAULT_NSF if USE_NSF_ONLY else DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver_param.select_reference_posture_method = METHOD_OF_SELECT_REFERENCE_POSTURE
    solver = Solver(solver_param=solver_param)

    (avg_l2, avg_ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        NUM_POSES,
        NUM_SOLS,
        std=STD,
        batch_size=BATCH_SIZE,
        success_threshold=SUCCESS_THRESHOLD,
    )  # type: ignore
    print(
        tabulate(
            [[avg_l2, np.rad2deg(avg_ang), avg_inference_time, success_rate]],
            headers=[
                "avg_l2 (m)",
                "avg_ang (deg)",
                "avg_inference_time (s)",
                f"success_rate ({METHOD_OF_SELECT_REFERENCE_POSTURE})",
            ],
        )
    )


def ikflow():
    set_seed()

    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    _, P = ik_solver.robot.sample_joint_angles_and_poses(n=NUM_POSES)
    l2 = np.zeros((NUM_SOLS, len(P)))
    ang = np.zeros((NUM_SOLS, len(P)))
    J = torch.empty((NUM_SOLS, len(P), 7), dtype=torch.float32, device="cpu")
    begin = time.time()
    if NUM_POSES < NUM_SOLS:
        for i in trange(NUM_POSES):
            J[:, i, :] = ik_solver.solve(
                P[i],
                n=NUM_SOLS,
                latent_scale=STD,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2[:, i], ang[:, i] = solution_pose_errors(
                ik_solver.robot, J[:, i, :], P[i]
            )
    else:
        for i in trange(NUM_SOLS):
            J[i] = ik_solver.solve_n_poses(
                P, latent_scale=STD, refine_solutions=False, return_detailed=False
            ).cpu()
            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J[i], P)
    avg_inference_time = round((time.time() - begin) / NUM_POSES, 3)

    print(
        tabulate(
            [[l2.mean(), np.rad2deg(ang.mean()), avg_inference_time]],
            headers=["avg_l2", "avg_ang", "avg_inference_time"],
        )
    )


if __name__ == "__main__":
    paik()
    ikflow()
