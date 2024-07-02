# Import required packages
import time
import numpy as np
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import PANDA_NSF, PANDA_PAIK

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from common.config import Config_IKP
from common.display import display_ikp

from common.config import Config_Posture
from common.display import display_posture
from common.evaluate import compute_distance_J

from common.config import Config_Diversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import (
    mmd_evaluate_multiple_poses,
    make_batches,
    batches_back_to_array,
)


def ikp():
    set_seed()
    config = Config_IKP()
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
                P,
                latent_scale=config.std,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()
            l2[i], ang[i] = solution_pose_errors(ik_solver.robot, J[i], P)
    avg_inference_time = round((time.time() - begin) / config.num_poses, 3)

    display_ikp(l2.mean(), ang.mean(), avg_inference_time)


def posture(config: Config_Posture):
    set_seed()
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


def diversity(config: Config_Diversity, solver: Any, std: float, P: np.ndarray):
    assert P.shape[:2] == (config.num_poses, config.num_sols)
    P = P.reshape(-1, P.shape[-1])
    P = make_batches(P, config.batch_size)  # type: ignore
    J_hat = batches_back_to_array(
        [
            solver.solve_n_poses(batch_P, latent_scale=std).cpu().numpy()
            for batch_P in tqdm(P)
        ]
    )
    J_hat = np.expand_dims(J_hat, axis=0)
    assert J_hat.shape == (
        1,
        config.num_poses * config.num_sols,
        J_hat.shape[-1],
    ), f"Expected: {(1, config.num_poses * config.num_sols, J_hat.shape[-1])}, Got: {J_hat.shape}"
    return J_hat


def ikflow(config: Config_Diversity, solver: Solver):
    set_seed()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    iterate_over_base_stds(config, "ikflow", ik_solver, solver, ikflow_solve)


def iterate_over_base_stds(
    config: Config_Diversity,
    iksolver_name: str,
    solver: Any,
    paik_solver: Solver,
    solve_fn,
):
    l2_mean = np.empty((len(config.base_stds)))
    ang_mean = np.empty((len(config.base_stds)))
    mmd_mean = np.empty((len(config.base_stds)))
    P, J_ground_truth = load_poses_and_numerical_ik_sols(config.record_dir)
    P = np.expand_dims(P, axis=1).repeat(config.num_sols, axis=1)
    J_hat_base_stds = np.empty(
        (
            len(config.base_stds),
            config.num_poses,
            config.num_sols,
            J_ground_truth.shape[-1],
        )
    )

    for i, std in enumerate(config.base_stds):
        J_hat = solve_fn(config, solver, std, P)

        l2, ang = paik_solver.evaluate_pose_error_J3d_P2d(
            J_hat, P.reshape(-1, P.shape[-1]), return_all=True
        )
        J_hat = J_hat.reshape(config.num_poses, config.num_sols, -1)
        l2_mean[i] = l2.mean()
        ang_mean[i] = ang.mean()
        mmd_mean[i] = mmd_evaluate_multiple_poses(
            J_hat, J_ground_truth, num_poses=config.num_poses
        )
        J_hat_base_stds[i] = J_hat
        assert not np.isnan(mmd_mean[i])

    save_diversity(
        config.record_dir,
        iksolver_name,
        J_hat_base_stds,
        l2_mean,
        ang_mean,
        mmd_mean,
        config.base_stds,
    )


if __name__ == "__main__":
    pass
