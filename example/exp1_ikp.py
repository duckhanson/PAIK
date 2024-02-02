# Import required packages
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
from tqdm import trange
from paik.solver import Solver
from paik.settings import (
    DEFAULT_NSF,
    DEFULT_SOLVER,
)
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

TEST_PAFIK = True
TEST_IKFLOW = True
NUM_POSES = 1000  # 100
NUM_SOLS = 1000  # 1000
BATCH_SIZE = 5000
SUCCESS_THRESHOLD = (5e-3, 2)
USE_NSF_ONLY = False
METHOD_OF_SELECT_REFERENCE_POSTURE = "knn"


def ikp(test_pafik: bool, test_ikflow: bool):
    solver_param = DEFAULT_NSF if USE_NSF_ONLY else DEFULT_SOLVER
    solver_param.method_of_select_reference_posture = METHOD_OF_SELECT_REFERENCE_POSTURE
    solver = Solver(solver_param=solver_param)

    if test_pafik:
        solver.shrink_ratio = 0.25
        # avg_l2, avg_ang, avg_inference_time, success_rate = solver.random_sample_solutions_with_evaluation(NUM_POSES, NUM_SOLS, success_threshold=SUCCESS_THRESHOLD)  # type: ignore
        (
            avg_l2,
            avg_ang,
            avg_inference_time,
            success_rate,
        ) = solver.random_sample_solutions_with_evaluation_loop(
            NUM_POSES,
            NUM_SOLS,
            batch_size=BATCH_SIZE,
            success_threshold=SUCCESS_THRESHOLD,
        )  # type: ignore
        print(
            tabulate(
                [
                    [
                        avg_l2,
                        np.rad2deg(avg_ang),
                        avg_inference_time,
                        success_rate,
                    ]
                ],
                headers=[
                    "avg_l2",
                    "avg_ang",
                    "avg_inference_time",
                    f"success_rate ({METHOD_OF_SELECT_REFERENCE_POSTURE})",
                ],
            )
        )

    if test_ikflow:
        set_seed()

        # Build IKFlowSolver and set weights
        ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

        _, P, _ = solver.get_random_JPF(NUM_POSES)  # type: ignore

        l2 = np.zeros((len(P), NUM_SOLS))
        ang = np.zeros((len(P), NUM_SOLS))
        J = torch.empty((NUM_SOLS, len(P), 7), dtype=torch.float32, device="cpu")
        begin = time.time()
        for i in trange(NUM_POSES):
            # (
            #     _,
            #     l2[i],
            #     ang[i],
            #     _,
            #     _,
            #     _,
            # ) = ik_solver.solve(
            #     P[i], n=NUM_SOLS, refine_solutions=False, return_detailed=True
            # )  # type: ignore

            J[:, i, :] = ik_solver.solve(
                P[i], n=NUM_SOLS, refine_solutions=False, return_detailed=False
            ).cpu()  # type: ignore

            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J[:, i, :], P[i])
        avg_inference_time = round((time.time() - begin) / NUM_POSES, 3)

        print(
            tabulate(
                [[l2.mean(), np.rad2deg(ang.mean()), avg_inference_time]],
                headers=["avg_l2", "avg_ang", "avg_inference_time"],
            )
        )


if __name__ == "__main__":
    ikp(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)
