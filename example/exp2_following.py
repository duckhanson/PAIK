# Import required packages
import numpy as np
import pandas as pd
from time import time
from tabulate import tabulate

from paik.solver import max_joint_angle_change
from paik.settings import DEFAULT_SOLVER_PARAM_M7_NORM
from paik.follower import PathFollower

import torch
from tqdm import trange
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

TEST_PAFIK = True
TEST_IKFLOW = True
LOAD_TIME = "1124134231"
NUM_TRAJECTORIES = 500
DDJC_THRES = (40, 50, 60, 70, 80, 90, 100)


def path_following(test_pafik: bool, test_ikflow: bool):
    solver = PathFollower(solver_param=DEFAULT_SOLVER_PARAM_M7_NORM)

    J, P = solver.sample_Jtraj_Ppath(load_time=LOAD_TIME, num_steps=20)

    if test_pafik:
        begin_time = time()
        J_hat, l2_errs, ang_errs, mjac_arr, ddjc = solver.solve_path(
            J, P, num_traj=NUM_TRAJECTORIES, return_numpy=True, return_evaluation=True
        )
        end_time = time()

        df = pd.DataFrame(
            {
                "l2_errs": l2_errs,
                "ang_errs": np.rad2deg(ang_errs),
                "mjac": mjac_arr,
                "ddjc": np.rad2deg(ddjc),
            }
        )

        table = np.empty((len(DDJC_THRES), 3))
        for i, thres in enumerate(DDJC_THRES):
            sr = df.query(f"ddjc < {thres}")["ddjc"].count() / df.shape[0]
            table[i] = (thres, sr, round(1 / sr)) if sr != 0 else (thres, sr, np.inf)
        print(tabulate(table, headers=["ddjc", "success rate", "min num sols"]))
        print(
            f"avg_inference_time: {round((end_time - begin_time) / NUM_TRAJECTORIES, 3)}"
        )
        print(df.describe())

    if test_ikflow:
        set_seed()

        # Build IKFlowSolver and set weights
        ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

        F_l2_err = np.empty((NUM_TRAJECTORIES,))
        F_ang_err = np.empty((NUM_TRAJECTORIES,))
        F_mjac = np.empty((NUM_TRAJECTORIES,))
        F_ddjc = np.empty((NUM_TRAJECTORIES,))
        F_runtime = np.empty((NUM_TRAJECTORIES))

        shape = (P.shape[0], ik_solver._network_width)
        begin_time = time()
        for i in trange(NUM_TRAJECTORIES):
            latent = torch.from_numpy(
                np.tile(np.random.randn(1, shape[1]), (shape[0], 1)).astype(np.float32)
            ).to("cuda")
            sol, l2, ang, _, _, runtime = ik_solver.solve_n_poses(
                P, latent=latent, refine_solutions=False, return_detailed=True
            )  # type: ignore
            sol = sol.cpu().numpy()
            F_l2_err[i] = l2.mean()
            F_ang_err[i] = ang.mean()
            F_mjac[i] = max_joint_angle_change(sol)  # type: ignore
            F_ddjc[i] = np.linalg.norm(sol - J, axis=-1).mean()
            F_runtime[i] = runtime
        end_time = time()
        F_df = pd.DataFrame(
            {
                "l2_err": F_l2_err,
                "ang_err": np.rad2deg(F_ang_err),
                "mjac": F_mjac,
                "ddjc": np.rad2deg(F_ddjc),
            }
        )

        table = np.empty((len(DDJC_THRES), 3))
        for i, thres in enumerate(DDJC_THRES):
            sr = F_df.query(f"ddjc < {thres}")["ddjc"].count() / F_df.shape[0]
            table[i] = (thres, sr, round(1 / sr)) if sr != 0 else (thres, sr, np.inf)
        print(tabulate(table, headers=["ddjc", "success rate", "min num sols"]))
        print(
            f"avg_inference_time: {round((end_time - begin_time) / NUM_TRAJECTORIES, 3)}"
        )
        print(F_df.describe())


if __name__ == "__main__":
    path_following(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)
