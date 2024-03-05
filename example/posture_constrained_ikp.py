# Import required packages
import numpy as np
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import DEFAULT_NSF, DEFULT_SOLVER

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from common.config import ConfigPosture
from common.display import display_posture
from common.evaluate import compute_distance_J


def paik():
    config = ConfigPosture()

    solver_param = DEFAULT_NSF if config.use_nsf_only else DEFULT_SOLVER
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)

    J, P = solver.robot.sample_joint_angles_and_poses(n=config.num_poses)

    # Data Preprocessing
    F = solver.F[solver.J_knn.kneighbors(J, return_distance=False).flatten()]

    # Begin inference
    J_hat = solver.solve_batch(
        P, F, num_sols=config.num_sols, batch_size=config.batch_size, verbose=True
    )

    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
    # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # J.shape = (num_poses, num_dofs or n)
    distance_J = compute_distance_J(J_hat, J)
    display_posture(
        config.record_dir,
        "paik",
        l2,
        ang,
        distance_J,
        config.success_distance_thresholds,
    )


def ikflow():
    set_seed()
    config = ConfigPosture()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    J, P = ik_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    l2 = np.zeros((config.num_sols, config.num_poses))
    ang = np.zeros((config.num_sols, config.num_poses))
    J_hat = torch.empty(
        (config.num_sols, config.num_poses, ik_solver.robot.n_dofs),
        dtype=torch.float32,
        device="cpu",
    )
    if config.num_poses < config.num_sols:
        for i in trange(config.num_poses):
            J_hat[:, i, :] = ik_solver.solve(
                P[i],
                n=config.num_sols,
                latent_scale=config.std,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2[:, i], ang[:, i] = solution_pose_errors(
                ik_solver.robot, J_hat[:, i, :], P[i]
            )
    else:
        for i in trange(config.num_sols):
            J_hat[i] = ik_solver.solve_n_poses(
                P,
                latent_scale=config.std,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()
            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J_hat[i], P)
    # J_hat.shape = (num_sols, num_poses, num_dofs or n)
    # J.shape = (num_poses, num_dofs or n)
    distance_J = compute_distance_J(J_hat, J)
    display_posture(
        config.record_dir,
        "ikflow",
        l2.flatten(),
        ang.flatten(),
        distance_J.flatten(),
        config.success_distance_thresholds,
    )


if __name__ == "__main__":
    paik()
    ikflow()
