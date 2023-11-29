# Import required packages
import numpy as np
import pandas as pd
from tabulate import tabulate
from paik.solver import Solver
from paik.settings import DEFAULT_SOLVER_PARAM_M7_NORM
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

TEST_PAFIK = True
TEST_IKFLOW = True
NUM_POSES = 100  # 100
NUM_SOLS = 1000  # 1000


def ikp(test_pafik: bool, test_ikflow: bool):
    solver = Solver(solver_param=DEFAULT_SOLVER_PARAM_M7_NORM)

    if test_pafik:
        solver.shrink_ratio = 0.25
        avg_l2_errs, avg_ang_errs, avg_inference_time = solver.random_sample_solutions_with_evaluation(NUM_POSES, NUM_SOLS, return_time=True)  # type: ignore

        print(
            tabulate(
                [[avg_l2_errs, avg_ang_errs, avg_inference_time]],
                headers=["avg_l2_errs", "avg_ang_errs", "avg_inference_time"],
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
                [[l2_errs.mean(), ang_errs.mean(), time_diffs.mean()]],
                headers=["avg_l2_errs", "avg_ang_errs", "avg_inference_time"],
            )
        )


if __name__ == "__main__":
    ikp(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)