# Import required packages
import numpy as np
import pandas as pd
from time import time
from tabulate import tabulate

from paik.settings import (
    DEFAULT_SOLVER_PARAM_M7_NORM,
    DEFAULT_SOLVER_PARAM_M7_DISABLE_POSTURE_FEATURES,
    DEFAULT_SOLVER_PARAM_M7_EXTRACT_FROM_C_SPACE,
)
from paik.follower import PathFollower, max_joint_angle_change

import torch
from tqdm import trange
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver

TEST_PAFIK = True
TEST_IKFLOW = True
DISABLE_POSTURE_FEATURE = False
EXTRACT_POSTURE_FEATURE_FROM_C_SPACE = True
LOAD_TIME = ""  # 0131005046
NUM_STEPS = 10
NUM_TRAJECTORIES = 1000
NUM_SOLS = 1
DDJC_THRES = (40, 80, 120)


def path_following(test_pafik: bool, test_ikflow: bool):
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
    solver = PathFollower(solver_param=solver_param)

    # J, P = solver.sample_Jtraj_Ppath(load_time=LOAD_TIME, num_steps=NUM_STEPS)
    J, P = solver.sample_Jtraj_Ppath_multiple_trajectories(
        num_steps=NUM_STEPS, num_traj=NUM_TRAJECTORIES
    )
    print(J.shape, P.shape)
    # if test_pafik:
    #     begin_time = time()
    #     J_hat, l2_errs, ang_errs, mjac_arr, ddjc = solver.solve_path(
    #         J, P, num_traj=NUM_TRAJECTORIES, return_numpy=True, return_evaluation=True
    #     )
    #     end_time = time()

    #     df = pd.DataFrame(
    #         {
    #             "l2_errs": l2_errs,
    #             "ang_errs": np.rad2deg(ang_errs),
    #             "mjac": mjac_arr,
    #             "ddjc": np.rad2deg(ddjc),
    #         }
    #     )

    #     table = np.empty((len(DDJC_THRES), 3))
    #     for i, thres in enumerate(DDJC_THRES):
    #         sr = df.query(f"ddjc < {thres}")["ddjc"].count() / df.shape[0]
    #         table[i] = (thres, sr, round(1 / sr)) if sr != 0 else (thres, sr, np.inf)
    #     print(tabulate(table, headers=["ddjc", "success rate", "min num sols"]))
    #     print(
    #         f"avg_inference_time: {round((end_time - begin_time) / NUM_TRAJECTORIES, 3)}"
    #     )
    #     print(df.describe())

    # if test_ikflow:
    #     set_seed()

    #     # Build IKFlowSolver and set weights
    #     ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

    #     F_l2_err = np.empty((NUM_TRAJECTORIES,))
    #     F_ang_err = np.empty((NUM_TRAJECTORIES,))
    #     F_mjac = np.empty((NUM_TRAJECTORIES,))
    #     F_ddjc = np.empty((NUM_TRAJECTORIES,))
    #     F_runtime = np.empty((NUM_TRAJECTORIES))

    #     shape = (P.shape[0], ik_solver._network_width)
    #     begin_time = time()
    #     for i in trange(NUM_TRAJECTORIES):
    #         latent = torch.from_numpy(
    #             np.tile(np.random.randn(1, shape[1]), (shape[0], 1)).astype(np.float32)
    #         ).to("cuda")
    #         sol, l2, ang, _, _, runtime = ik_solver.solve_n_poses(
    #             P, latent=latent, refine_solutions=False, return_detailed=True
    #         )  # type: ignore
    #         sol = sol.cpu().numpy()
    #         F_l2_err[i] = l2.mean()
    #         F_ang_err[i] = ang.mean()
    #         F_mjac[i] = max_joint_angle_change(sol)  # type: ignore
    #         F_ddjc[i] = np.linalg.norm(sol - J, axis=-1).mean()
    #         F_runtime[i] = runtime
    #     end_time = time()
    #     F_df = pd.DataFrame(
    #         {
    #             "l2_err": F_l2_err,
    #             "ang_err": np.rad2deg(F_ang_err),
    #             "mjac": F_mjac,
    #             "ddjc": np.rad2deg(F_ddjc),
    #         }
    #     )

    #     table = np.empty((len(DDJC_THRES), 3))
    #     for i, thres in enumerate(DDJC_THRES):
    #         sr = F_df.query(f"ddjc < {thres}")["ddjc"].count() / F_df.shape[0]
    #         table[i] = (thres, sr, round(1 / sr)) if sr != 0 else (thres, sr, np.inf)
    #     print(tabulate(table, headers=["ddjc", "success rate", "min num sols"]))
    #     print(
    #         f"avg_inference_time: {round((end_time - begin_time) / NUM_TRAJECTORIES, 3)}"
    #     )
    #     print(F_df.describe())


def path_following_multiple_trajectory(test_pafik: bool, test_ikflow: bool):
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
    solver = PathFollower(solver_param=solver_param)

    J, P = solver.sample_Jtraj_Ppath_multiple_trajectories(
        num_steps=NUM_STEPS, num_traj=NUM_TRAJECTORIES
    )

    if test_pafik:
        l2 = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        ang = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        mjac = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        ddjc = np.empty((NUM_TRAJECTORIES, NUM_SOLS))

        begin_time = time()
        for i in trange(NUM_TRAJECTORIES):
            _, l2[i], ang[i], mjac[i], ddjc[i] = solver.solve_path(
                np.atleast_2d(J[i]),
                np.atleast_2d(P[i]),
                num_sols=NUM_SOLS,
                return_numpy=True,
                return_evaluation=True,
            )
        avg_inference_time = round((time() - begin_time) / NUM_TRAJECTORIES, 3)

        df = pd.DataFrame(
            {
                "l2": l2.flatten(),
                "ang": np.rad2deg(ang.flatten()),
                "mjac": mjac.flatten(),
                "ddjc": np.rad2deg(ddjc.flatten()),
            }
        )

        print(
            tabulate(
                [
                    (thres, df.query(f"ddjc < {thres}")["ddjc"].count() / df.shape[0])
                    for thres in DDJC_THRES
                ],
                headers=["ddjc", "success rate"],
            )
        )
        print(df.describe())
        print(f"avg_inference_time: {avg_inference_time}")

    if test_ikflow:
        set_seed()

        # Build IKFlowSolver and set weights
        ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

        F_l2_err = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        F_ang_err = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        F_mjac = np.empty((NUM_TRAJECTORIES, NUM_SOLS))
        F_ddjc = np.empty((NUM_TRAJECTORIES, NUM_SOLS))

        begin_time = time()
        for i in trange(NUM_TRAJECTORIES):
            for j in range(NUM_SOLS):
                sol, l2, ang, _, _, _ = ik_solver.solve_n_poses(
                    P[i],
                    latent_scale=0.25,
                    refine_solutions=False,
                    return_detailed=True,
                )  # type: ignore
                sol = sol.cpu().numpy()
                F_l2_err[i, j] = l2.mean()
                F_ang_err[i, j] = ang.mean()
                F_mjac[i, j] = max_joint_angle_change(sol)  # type: ignore
                F_ddjc[i, j] = np.linalg.norm(sol - J[i], axis=-1).mean()
        avg_inference_time = round((time() - begin_time) / NUM_TRAJECTORIES, 3)

        F_df = pd.DataFrame(
            {
                "l2_err": F_l2_err.flatten(),
                "ang_err": np.rad2deg(F_ang_err.flatten()),
                "mjac": F_mjac.flatten(),
                "ddjc": np.rad2deg(F_ddjc.flatten()),
            }
        )

        print(
            tabulate(
                [
                    (
                        thres,
                        F_df.query(f"ddjc < {thres}")["ddjc"].count() / F_df.shape[0],
                    )
                    for thres in DDJC_THRES
                ],
                headers=["ddjc", "success rate"],
            )
        )
        print(F_df.describe())
        print(f"avg_inference_time: {avg_inference_time}")


if __name__ == "__main__":
    path_following_multiple_trajectory(test_pafik=TEST_PAFIK, test_ikflow=TEST_IKFLOW)
