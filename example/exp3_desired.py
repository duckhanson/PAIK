import numpy as np
import pandas as pd
from paik.follower import PathFollower
from paik.settings import (
    DEFAULT_SOLVER_PARAM_M7_NORM,
    DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE,
)
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

TEST_PAFIK = True
TEST_IKFLOW = True
NUM_IMITATION_DATA = 100
NUM_TARGET_DATA = 100
NUM_SOLUTIONS = 100


def desired(test_pafik: bool, test_ikflow: bool):
    solver = PathFollower(solver_param=DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE)
    solver.shrink_ratio = 0.0
    J_im, P_im, F_im = solver.get_random_JPF(num_samples=NUM_IMITATION_DATA)
    J_ta, P_ta, F_ta = solver.get_random_JPF(num_samples=NUM_TARGET_DATA)

    if test_pafik:
        l2_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))
        ang_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))
        config_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))

        for i, (j_im, f_im) in enumerate(zip(J_im, F_im)):
            # F = np.full_like(F_ta, *f_im)
            j_im = np.tile(j_im, (NUM_TARGET_DATA, 1))
            F = solver._F[solver.J_knn.kneighbors(np.atleast_2d(j_im), return_distance=False).flatten()]  # type: ignore
            J = solver.solve(P=P_ta, F=F, num_sols=NUM_SOLUTIONS, return_numpy=True)
            l2_errs[i], ang_errs[i] = solver.evaluate_solutions(
                J, P_ta, return_col=True
            )
            config_errs[i] = np.linalg.norm(J - j_im, axis=2).mean(axis=0)

        df = pd.DataFrame(
            {
                "l2_err": l2_errs.flatten(),
                "ang_err": np.rad2deg(ang_errs.flatten()),
                "config_err": np.rad2deg(config_errs.flatten()),
            }
        )

        print(df.describe())

    if test_ikflow:
        set_seed()

        # Build IKFlowSolver and set weights
        ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

        # -> unrefined solutions
        l2_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))
        ang_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))
        config_errs = np.zeros((NUM_IMITATION_DATA, NUM_TARGET_DATA))

        for i, p in enumerate(P_ta):
            (
                solutions,
                l2_errors,
                angular_errors,
                _,
                _,
                _,  # type: ignore
            ) = ik_solver.solve(
                p, n=NUM_IMITATION_DATA, refine_solutions=False, return_detailed=True
            )  # type: ignore

            l2_errs[:, i] = l2_errors
            ang_errs[:, i] = angular_errors

            for j, j_im in enumerate(J_im):
                config_errs[j, i] = np.linalg.norm(
                    solutions.detach().cpu().numpy() - j_im, axis=1
                ).mean()

        df = pd.DataFrame(
            {
                "l2_err": l2_errs.flatten(),
                "ang_err": np.rad2deg(ang_errs.flatten()),
                "config_err": np.rad2deg(config_errs.flatten()),
            }
        )
        print(df.describe())


if __name__ == "__main__":
    desired(TEST_PAFIK, TEST_IKFLOW)
