# Import required packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from paik.solver import Solver
from paik.settings import (
    DEFAULT_NSF,
    DEFULT_SOLVER,
)

from common.config import ConfigDiversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import mmd_evaluate_multiple_poses

from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver


def get_numerical_ik_sols(pose, num_seeds):
    seeds, _ = solver._robot.sample_joint_angles_and_poses(
        n=num_seeds, return_torch=False
    )
    
    numerical_ik_sols = np.empty((num_seeds, solver.n))
    for i, seed in enumerate(seeds):
        numerical_ik_sols[i] = solver.robot.inverse_kinematics_klampt(pose=pose, seed=seed)
    return numerical_ik_sols


def klampt_numerical_ik_solver(config: ConfigDiversity, solver: Solver):
    _, P = solver._robot.sample_joint_angles_and_poses(
        n=config.num_poses, return_torch=False
    )

    # shape: (num_poses, num_sols, num_dofs or n)
    J_ground_truth = np.asarray([get_numerical_ik_sols(p, config.num_sols) for p in tqdm(P)])

    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_ground_truth.transpose(1, 0, 2), P, return_all=True)
    df = pd.DataFrame({"l2": l2, "ang": ang})
    print(df.describe())
    
    # Save to repeat the same experiment on NODEIK
    np.save(f"{config.record_dir}/numerical_ik_sols.npy", J_ground_truth)
    np.save(f"{config.record_dir}/poses.npy", P)


def paik(config: ConfigDiversity, solver: Solver):
    l2_mean = np.empty((len(config.base_stds)))
    ang_mean = np.empty((len(config.base_stds)))
    mmd_mean = np.empty((len(config.base_stds)))
    J_hat_paik = np.empty(
        (len(config.base_stds), config.num_poses, config.num_sols, solver.n)
    )
    P, J_ground_truth = load_poses_and_numerical_ik_sols(config.record_dir)

    for i, std in enumerate(config.base_stds):
        solver.base_std = std
        P_expand_dim = (
            np.expand_dims(P, axis=1)
            .repeat(config.num_sols, axis=1)
            .reshape(-1, P.shape[-1])
        )

        F = solver.F[
            solver.P_knn.kneighbors(
                np.atleast_2d(P), n_neighbors=config.num_sols, return_distance=False
            ).flatten()
        ]
        J_hat = solver.solve_batch(P_expand_dim, F, 1)

        l2, ang = solver.evaluate_pose_error_J3d_P2d(
            J_hat, P_expand_dim, return_all=True
        )
        J_hat = J_hat.reshape(config.num_poses, config.num_sols, -1)
        J_hat_paik[i] = J_hat
        l2_mean[i] = l2.mean()
        ang_mean[i] = ang.mean()
        mmd_mean[i] = mmd_evaluate_multiple_poses(
            J_hat, J_ground_truth, num_poses=config.num_poses
        )
        assert not np.isnan(mmd_mean[i])

    save_diversity(
        config.record_dir,
        "paik",
        J_hat_paik,
        l2_mean,
        ang_mean,
        mmd_mean,
        config.base_stds,
    )


def ikflow(config: ConfigDiversity, solver: Solver):
    set_seed()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    l2_flow = np.empty((len(config.base_stds)))
    ang_flow = np.empty((len(config.base_stds)))
    mmd_flow = np.empty((len(config.base_stds)))
    J_hat_ikflow = np.empty(
        (len(config.base_stds), config.num_poses, config.num_sols, solver.n)
    )

    P, J_ground_truth = load_poses_and_numerical_ik_sols(config.record_dir)

    for i, std in enumerate(config.base_stds):
        J_flow = np.array(
            [
                ik_solver.solve(p, n=config.num_sols, latent_scale=std).cpu().numpy()
                for p in P
            ]
        )  # (num_sols, num_poses, n)
        J_hat_ikflow[i] = J_flow

        l2, ang = solver.evaluate_pose_error_J3d_P2d(
            J_flow.transpose(1, 0, 2), P, return_all=True
        )
        l2_flow[i] = l2.mean()
        ang_flow[i] = ang.mean()
        mmd_flow[i] = mmd_evaluate_multiple_poses(
            J_flow, J_ground_truth, num_poses=config.num_poses
        )

    save_diversity(
        config.record_dir,
        "ikflow",
        J_hat_ikflow,
        l2_flow,
        ang_flow,
        mmd_flow,
        config.base_stds,
    )


if __name__ == "__main__":
    config = ConfigDiversity()
    solver_param = DEFULT_SOLVER
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)

    klampt_numerical_ik_solver(config, solver)
    paik(config, solver)
    ikflow(config, solver)
