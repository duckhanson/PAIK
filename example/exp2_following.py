# Import required packages
import numpy as np
import pandas as pd
from time import time
from tabulate import tabulate
from pafik.settings import DEFAULT_NSF, DEFULT_SOLVER
from pafik.follower import PathFollower, max_joint_angle_change
import torch
from tqdm import trange


WORKDIR = "/home/luca/example_package/pafik"
TEST_PAFIK = True
TEST_IKFLOW = True
USE_NSF_ONLY = False
LOAD_TIME = ""  # 0131005046
NUM_STEPS = 20
NUM_TRAJECTORIES = 10000
NUM_SOLS = 1
STD = 0.01
DDJC_THRES = (40, 80, 120)


def path_following_multiple_trajectory(test_pafik: bool, test_ikflow: bool):
    solver_param = DEFAULT_NSF if USE_NSF_ONLY else DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver = PathFollower(solver_param=solver_param)

    Jt, Pt = solver.sample_multiple_Jtraj_and_Ppath(
        num_steps=NUM_STEPS, num_traj=NUM_TRAJECTORIES
    )

    begin_time = time()
    J = Jt.reshape(-1, Jt.shape[-1])
    P = Pt.reshape(-1, Pt.shape[-1])
    solver.base_std = STD
    J_hat = solver.solve_batch(
        P,
        solver._F[solver.J_knn.kneighbors(J, return_distance=False).flatten()],
        num_sols=1,
    )  # type: ignore
    l2, ang = solver.evaluate_pose_error(J_hat, P, return_all=True)
    ddjc = np.linalg.norm(
        J_hat - J, axis=-1).reshape(NUM_TRAJECTORIES, NUM_STEPS)
    mjac = np.array(
        [
            max_joint_angle_change(qs)
            for qs in J_hat.reshape(NUM_TRAJECTORIES, NUM_STEPS, -1)
        ]
    )

    avg_inference_time = round((time() - begin_time) / NUM_TRAJECTORIES, 3)
    l2 = l2.reshape(NUM_TRAJECTORIES, NUM_STEPS).mean(axis=-1)
    ang = ang.reshape(NUM_TRAJECTORIES, NUM_STEPS).mean(axis=-1)
    ddjc = ddjc.mean(axis=-1)
    df = pd.DataFrame(
        {"l2": l2, "ang": np.rad2deg(
            ang), "mjac": mjac, "ddjc": np.rad2deg(ddjc)}
    )

    print(
        tabulate(
            [
                (thres, df.query(f"ddjc < {thres}")
                 ["ddjc"].count() / df.shape[0])
                for thres in DDJC_THRES
            ],
            headers=["ddjc", "success rate"],
        )
    )
    print(df.describe())
    print(f"avg_inference_time: {avg_inference_time}")


if __name__ == "__main__":
    path_following_multiple_trajectory()
