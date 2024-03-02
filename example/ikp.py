# Import required packages
import dis
import time
import numpy as np
from tabulate import tabulate
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import DEFAULT_NSF, DEFULT_SOLVER

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from common.config import ConfigIKP
from common.display import display_ikp


def paik():
    config = ConfigIKP()
    solver_param = DEFAULT_NSF if config.use_nsf_only else DEFULT_SOLVER
    solver_param.workdir = config.workdir
    solver_param.select_reference_posture_method = config.method_of_select_reference_posture
    solver = Solver(solver_param=solver_param)

    (l2, ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
    )  # type: ignore
    display_ikp(l2, ang, avg_inference_time)
    print(
        f"success rate {config.method_of_select_reference_posture}: {success_rate}")


def ikflow():
    set_seed()
    config = ConfigIKP()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    _, P = ik_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    l2 = np.zeros((config.num_sols, len(P)))
    ang = np.zeros((config.num_sols, len(P)))
    J = torch.empty((config.num_sols, len(P), 7),
                    dtype=torch.float32, device="cpu")
    begin = time.time()
    if config.num_poses < config.num_sols:
        for i in trange(config.num_poses):
            J[:, i, :] = ik_solver.solve(
                P[i],
                n=config.num_sols,
                latent_scale=config.std,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2[:, i], ang[:, i] = solution_pose_errors(
                ik_solver.robot, J[:, i, :], P[i]
            )
    else:
        for i in trange(config.num_sols):
            J[i] = ik_solver.solve_n_poses(
                P, latent_scale=config.std, refine_solutions=False, return_detailed=False
            ).cpu()
            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J[i], P)
    avg_inference_time = round((time.time() - begin) / config.num_poses, 3)

    display_ikp(l2.mean(), ang.mean(), avg_inference_time)


if __name__ == "__main__":
    paik()
    ikflow()
