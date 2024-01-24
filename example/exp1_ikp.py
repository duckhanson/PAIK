# Import required packages
import numpy as np
import pandas as pd
from tabulate import tabulate
from paik.solver import Solver
from paik.settings import (
    DEFAULT_SOLVER_PARAM_M7_NORM,
    DEFAULT_SOLVER_PARAM_M7_DISABLE_POSTURE_FEATURES,
    DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE,
)
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

TEST_PAFIK = True
TEST_IKFLOW = False
NUM_POSES = 10000  # 100
NUM_SOLS = 1000  # 1000
SUCCESS_THRESHOLD = (5e-3, 2)
DISABLE_POSTURE_FEATURE = False
EXTRACT_POSTURE_FEATURE_FROM_C_SPACE = True
METHOD_OF_SELECT_REFERENCE_POSTURE = "pick"

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
        avg_l2_errs, avg_ang_errs, avg_inference_time, success_rate = solver.random_sample_solutions_with_evaluation(NUM_POSES, NUM_SOLS, return_time=True, return_success_rate=True, success_threshold=SUCCESS_THRESHOLD)  # type: ignore
        print(
            tabulate(
                [
                    [
                        avg_l2_errs,
                        np.rad2deg(avg_ang_errs),
                        avg_inference_time,
                        success_rate,
                    ]
                ],
                headers=[
                    "avg_l2_errs",
                    "avg_ang_errs",
                    "avg_inference_time",
                    "success_rate",
                ],
            )
        )

    if test_ikflow:
        set_seed()

        # Build IKFlowSolver and set weights
        ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

        _, P, _ = solver.get_random_JPF(NUM_POSES)  # type: ignore

        # -> unrefined solutions
        l2_errs = np.zeros((len(P), NUM_SOLS))
        ang_errs = np.zeros((len(P), NUM_SOLS))

        time_diffs = np.zeros((len(P)))

        for i, p in enumerate(P):
            (
                _,
                l2_errs[i],
                ang_errs[i],
                _,
                _,
                time_diffs[i],  # type: ignore
            ) = ik_solver.solve(
                p, n=NUM_SOLS, refine_solutions=False, return_detailed=True
            )  # type: ignore

        print(
            tabulate(
                [[l2_errs.mean(), np.rad2deg(ang_errs.mean()), time_diffs.mean()]],
                headers=["avg_l2_errs", "avg_ang_errs", "avg_inference_time"],
            )
        )


if __name__ == "__main__":
    ikp(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)
