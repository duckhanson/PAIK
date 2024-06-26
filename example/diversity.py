# Import required packages
import time
from typing import Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from paik.solver import Solver
from paik.settings import (
    PANDA_NSF,
    PANDA_PAIK,
)

from common.config import ConfigDiversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import (
    mmd_evaluate_multiple_poses,
    make_batches,
    batches_back_to_array,
)

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver


def get_numerical_ik_sols(pose, num_seeds):
    seeds, _ = solver._robot.sample_joint_angles_and_poses(
        n=num_seeds, return_torch=False
    )

    numerical_ik_sols = np.empty((num_seeds, solver.n))
    for i, seed in enumerate(seeds):
        numerical_ik_sols[i] = solver.robot.inverse_kinematics_klampt(
            pose=pose, seed=seed
        )
    return numerical_ik_sols


def klampt_numerical_ik_solver(config: ConfigDiversity, solver: Solver):
    _, P = solver._robot.sample_joint_angles_and_poses(
        n=config.num_poses, return_torch=False
    )

    # shape: (num_poses, num_sols, num_dofs or n)
    begin = time.time()
    J_ground_truth = np.asarray(
        [get_numerical_ik_sols(p, config.num_sols) for p in tqdm(P)]
    )
    print(
        f"Time to solve {config.num_poses} poses: {time.time() - begin:.2f}s")

    l2, ang = solver.evaluate_pose_error_J3d_P2d(
        J_ground_truth.transpose(1, 0, 2), P, return_all=True
    )
    df = pd.DataFrame({"l2": l2, "ang": ang})
    print(df.describe())

    # Save to repeat the same experiment on NODEIK
    np.save(f"{config.record_dir}/numerical_ik_sols.npy", J_ground_truth)
    np.save(f"{config.record_dir}/poses.npy", P)


def paik_solve(config: ConfigDiversity, solver: Solver, std: float, P: np.ndarray):
    assert P.shape[:2] == (config.num_poses, config.num_sols)

    solver.base_std = std
    # shape: (num_poses * num_sols, n)
    F = solver.F[
        solver.P_knn.kneighbors(
            np.atleast_2d(P[:, 0]), n_neighbors=config.num_sols, return_distance=False
        ).flatten()
    ]

    # shape: (num_poses * num_sols, n)
    P = P.reshape(-1, P.shape[-1])
    J_hat = solver.solve_batch(P, F, 1)  # (1, num_poses * num_sols, n)
    assert J_hat.shape == (
        1,
        config.num_poses * config.num_sols,
        solver.n,
    ), f"Expected: {(1, config.num_poses * config.num_sols, solver.n)}, Got: {J_hat.shape}"
    return J_hat


def nsf_solve(config: ConfigDiversity, solver: Solver, std: float, P: np.ndarray):
    assert P.shape[:2] == (config.num_poses, config.num_sols)

    solver.base_std = std
    # shape: (num_poses * num_sols, n)
    F = solver.F[
        solver.P_knn.kneighbors(
            np.atleast_2d(P[:, 0]), n_neighbors=config.num_sols, return_distance=False
        ).flatten()
    ]

    # shape: (num_poses * num_sols, n)
    P = P.reshape(-1, P.shape[-1])
    J_hat = solver.solve_batch(P, F, 1)  # (1, num_poses * num_sols, n)
    assert J_hat.shape == (
        1,
        config.num_poses * config.num_sols,
        solver.n,
    ), f"Expected: {(1, config.num_poses * config.num_sols, solver.n)}, Got: {J_hat.shape}"
    return J_hat


def ikflow_solve(config: ConfigDiversity, solver: Any, std: float, P: np.ndarray):
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


def iterate_over_base_stds(
    config: ConfigDiversity,
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


def paik(config: ConfigDiversity, solver: Solver):
    iterate_over_base_stds(config, "paik", solver, solver, paik_solve)


def nsf(config: ConfigDiversity, solver: Solver):
    solver_param = PANDA_NSF
    solver_param.workdir = config.workdir
    nsf = Solver(solver_param=solver_param)
    iterate_over_base_stds(config, "nsf", nsf, solver, nsf_solve)


def ikflow(config: ConfigDiversity, solver: Solver):
    set_seed()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    iterate_over_base_stds(config, "ikflow", ik_solver, solver, ikflow_solve)


if __name__ == "__main__":
    config = ConfigDiversity()
    solver_param = PANDA_PAIK
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)
    config.date = "2024_03_04"
    # klampt_numerical_ik_solver(config, solver)
    # paik(config, solver)
    nsf(config, solver)
    # ikflow(config, solver)
