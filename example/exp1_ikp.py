# Import required packages
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import trange
from paik.solver import Solver
from paik.settings import (
    DEFAULT_SOLVER_PARAM_M7_NORM,
    DEFAULT_SOLVER_PARAM_M7_DISABLE_POSTURE_FEATURES,
    DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE,
)
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

TEST_PAFIK = True
TEST_IKFLOW = False
NUM_POSES = 1000  # 100
NUM_SOLS = 1000  # 1000
SUCCESS_THRESHOLD = (5e-3, 2)
DISABLE_POSTURE_FEATURE = True
EXTRACT_POSTURE_FEATURE_FROM_C_SPACE = False
METHOD_OF_SELECT_REFERENCE_POSTURE = "knn"


def ikp(test_pafik: bool, test_ikflow: bool):
    assert not (DISABLE_POSTURE_FEATURE and EXTRACT_POSTURE_FEATURE_FROM_C_SPACE)
    solver_param = (
        DEFAULT_SOLVER_PARAM_M7_DISABLE_POSTURE_FEATURES
        if DISABLE_POSTURE_FEATURE
        else DEFAULT_SOLVER_PARAM_M7_NORM
    )
    solver_param = (
        DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE
        if EXTRACT_POSTURE_FEATURE_FROM_C_SPACE
        else solver_param
    )
    solver_param.method_of_select_reference_posture = METHOD_OF_SELECT_REFERENCE_POSTURE
    solver = Solver(solver_param=solver_param)

    if test_pafik:
        solver.shrink_ratio = 0.25
        # avg_l2, avg_ang, avg_inference_time, success_rate = solver.random_sample_solutions_with_evaluation(NUM_POSES, NUM_SOLS, success_threshold=SUCCESS_THRESHOLD)  # type: ignore
        avg_l2, avg_ang, avg_inference_time, success_rate = solver.random_sample_solutions_with_evaluation_loop(NUM_POSES, NUM_SOLS, success_threshold=SUCCESS_THRESHOLD)  # type: ignore
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
        J = np.zeros((NUM_SOLS, len(P), 7))
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
            ).cpu().numpy()  # type: ignore
        avg_inference_time = round((time.time() - begin) / NUM_POSES, 3)

        l2, ang = solver.evaluate_solutions(J, P)  # type: ignore
        # l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J, P[i])

        print(
            tabulate(
                [[l2.mean(), np.rad2deg(ang.mean()), avg_inference_time]],
                headers=["avg_l2", "avg_ang", "avg_inference_time"],
            )
        )


if __name__ == "__main__":
    ikp(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)
