# Import required packages
import numpy as np
import pandas as pd
from time import time
from tabulate import tabulate
import torch
from tqdm import trange
from pafik.settings import DEFULT_SOLVER
from pafik.follower import PathFollower, max_joint_angle_change
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors


WORKDIR = "/home/luca/example_package/pafik"
LOAD_TIME = ""
NUM_STEPS = 20
NUM_TRAJECTORIES = 10000
NUM_SOLS = 1
STD = 0.01
DDJC_THRES = (40, 80, 120)


def path_following_multiple_trajectory():
    solver_param = DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver = PathFollower(solver_param=solver_param)

    Jt, Pt = solver.sample_multiple_Jtraj_and_Ppath(
        num_steps=NUM_STEPS, num_traj=NUM_TRAJECTORIES
    )

    set_seed()

    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")

    l2 = np.empty((NUM_TRAJECTORIES, NUM_STEPS))
    ang = np.empty((NUM_TRAJECTORIES, NUM_STEPS))
    mjac = np.empty((NUM_TRAJECTORIES))
    ddjc = np.empty((NUM_TRAJECTORIES, NUM_STEPS))

    begin_time = time()
    J_hat = torch.empty(
        (NUM_TRAJECTORIES, NUM_STEPS, Jt.shape[-1]), dtype=torch.float32, device="cpu"
    )
    for i in trange(NUM_STEPS):
        J_hat[:, i, :] = ik_solver.solve_n_poses(
            Pt[:, i], latent_scale=STD, refine_solutions=False, return_detailed=False
        ).cpu()  # type: ignore
        l2[:, i], ang[:, i] = solution_pose_errors(
            ik_solver.robot, J_hat[:, i, :], Pt[:, i]
        )

    ddjc = np.linalg.norm(J_hat - Jt, axis=-1)
    mjac = np.array([max_joint_angle_change(qs) for qs in J_hat])

    avg_inference_time = round((time() - begin_time) / NUM_TRAJECTORIES, 3)

    df = pd.DataFrame(
        {
            "l2": l2.mean(axis=1),
            "ang": np.rad2deg(ang.mean(axis=1)),
            "mjac": mjac,
            "ddjc": np.rad2deg(ddjc.mean(axis=1)),
        }
    )
    print(df.describe())

    print(
        tabulate(
            [
                (thres, df.query(f"ddjc < {thres}")["ddjc"].count() / df.shape[0])
                for thres in DDJC_THRES
            ],
            headers=["ddjc", "success rate"],
        )
    )
    print(f"avg_inference_time: {avg_inference_time}")


if __name__ == "__main__":
    path_following_multiple_trajectory()
