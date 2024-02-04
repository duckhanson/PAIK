# Import required packages
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
from tqdm import trange

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors


TEST_IKFLOW = True
NUM_POSES = 3000  # 100
NUM_SOLS = 400  # 1000
STD = 0.1


def ikp():
    set_seed()

    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    _, P = ik_solver.robot.sample_joint_angles_and_poses(
        n=NUM_POSES, return_torch=False
    )
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
