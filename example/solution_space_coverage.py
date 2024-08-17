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

from common.config import Config_Diversity
from common.file import save_diversity, load_poses_and_numerical_ik_sols
from common.evaluate import (
    mmd_evaluate_multiple_poses,
    make_batches,
    batches_back_to_array,
)


def numerical_inverse_kinematics_single(solver, pose, num_sols):
    seeds, _ = solver.robot.sample_joint_angles_and_poses(n=num_sols)

    ik_sols = np.empty((num_sols, solver.n))
    for i, seed in enumerate(seeds):
        ik_sols[i] = solver.robot.inverse_kinematics_klampt(
            pose=pose, seed=seed
        ) # type: ignore
    return ik_sols

def numerical_inverse_kinematics_batch(solver, P, num_sols):
    # return shape: (1, num_poses*num_sols, n)
    return np.asarray(
        [numerical_inverse_kinematics_single(solver, p, num_sols) for p in P]
    ).reshape(1, -1, solver.n)
    
def paik_batch(solver, P, num_sols, std=0.001):
    # P.shape = (num_poses, m)
    # return shape: (1, num_poses*num_sols, n)
    
    # shape: (num_poses, num_sols, m)
    P_num_sols = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
    solver.base_std = std
    
    # shape: (num_poses * num_sols, n)
    F = solver.F[
        solver.P_knn.kneighbors(
            np.atleast_2d(P_num_sols[:, 0]), n_neighbors=num_sols, return_distance=False
        ).flatten()
    ]

    # shape: (num_poses * num_sols, n)
    P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
    
    # shape: (1, num_poses * num_sols, n)
    J_hat = solver.solve_batch(P_num_sols, F, 1)
    
    # return shape: (1, num_poses*num_sols, n)
    return J_hat

def nsf_batch(solver, P, num_sols, std=0.001):
    # P.shape = (num_poses, m)
    # return shape: (1, num_poses*num_sols, n)
    assert solver.param.use_nsf_only == True, "Solver is not NSF"
    return paik_batch(solver, P, num_sols, std)

def random_ikp(solver: Solver, P, num_sols, solve_fn_batch, std=None):
    # shape: (num_poses, num_sols, num_dofs or n)
    num_poses = P.shape[0]
    begin = time.time()
    if std is None:
        J_hat = solve_fn_batch(solver, P, num_sols)
    else:
        J_hat = solve_fn_batch(solver, P, num_sols, std)

    l2, ang = solver.evaluate_pose_error_J3d_P2d(
        # input J.shape = (num_sols, num_poses, num_dofs or n)
        J_hat.reshape(num_poses, num_sols, solver.n).transpose(1, 0, 2), P, return_all=True
    )
    print(f"Time to solve {num_sols} solutions (avg over {num_poses} poses): {(time.time() - begin)/num_poses *1000:.1f}ms")
    df = pd.DataFrame({"l2": l2, "ang": ang})
    print(df.describe())

    # # Save to repeat the same experiment on NODEIK
    # np.save(f"{config.record_dir}/numerical_ik_sols.npy", J_ground_truth)
    # np.save(f"{config.record_dir}/poses.npy", P)
    return J_hat

# def iterate_over_num_sols(
#     config: Config_Diversity,
#     iksolver_name: str,
#     solver: Any,
#     paik_solver: Solver,
#     solve_fn,
# ):
#     num_sols_arr = np.arange(100, 1000, 200)
#     l2_mean = np.empty((len(num_sols_arr)))
#     ang_mean = np.empty((len(num_sols_arr)))
#     mmd_mean = np.empty((len(num_sols_arr)))

#     for i, num_sols in enumerate(num_sols_arr):
#         P = np.expand_dims(P, axis=1).repeat(num_sols, axis=1)
#         J_hat = solve_fn(config, solver, config.base_std, P)

#         l2, ang = paik_solver.evaluate_pose_error_J3d_P2d(
#             J_hat, P.reshape(-1, P.shape[-1]), return_all=True
#         )
#         J_hat = J_hat.reshape(config.num_poses, num_sols, -1)
#         l2_mean[i] = l2.mean()
#         ang_mean[i] = ang.mean()
#         mmd_mean[i] = mmd_evaluate_multiple_poses(
#             J_hat, J_ground_truth, num_poses=config.num_poses
#         )
#         J_hat_num_sols[i] = J_hat
#         assert not np.isnan(mmd_mean[i])

#     save_diversity(
#         config.record_dir,
#         iksolver_name,
#         J_hat_num_sols,
#         l2_mean,
#         ang_mean,
#         mmd_mean,
#         config.num_sols,
#     )
# )


# def iterate_over_base_stds(
#     config: Config_Diversity,
#     iksolver_name: str,
#     solver: Any,
#     paik_solver: Solver,
#     solve_fn,
# ):
#     l2_mean = np.empty((len(config.base_stds)))
#     ang_mean = np.empty((len(config.base_stds)))
#     mmd_mean = np.empty((len(config.base_stds)))
#     P, J_ground_truth = load_poses_and_numerical_ik_sols(config.record_dir)
#     P = np.expand_dims(P, axis=1).repeat(config.num_sols, axis=1)
#     J_hat_base_stds = np.empty(
#         (
#             len(config.base_stds),
#             config.num_poses,
#             config.num_sols,
#             J_ground_truth.shape[-1],
#         )
#     )

#     for i, std in enumerate(config.base_stds):
#         J_hat = solve_fn(config, solver, std, P)

#         l2, ang = paik_solver.evaluate_pose_error_J3d_P2d(
#             J_hat, P.reshape(-1, P.shape[-1]), return_all=True
#         )
#         J_hat = J_hat.reshape(config.num_poses, config.num_sols, -1)
#         l2_mean[i] = l2.mean()
#         ang_mean[i] = ang.mean()
#         mmd_mean[i] = mmd_evaluate_multiple_poses(
#             J_hat, J_ground_truth, num_poses=config.num_poses
#         )
#         J_hat_base_stds[i] = J_hat
#         assert not np.isnan(mmd_mean[i])

#     save_diversity(
#         config.record_dir,
#         iksolver_name,
#         J_hat_base_stds,
#         l2_mean,
#         ang_mean,
#         mmd_mean,
#         config.base_stds,
#     )


# def paik(config: Config_Diversity, solver: Solver):
#     iterate_over_base_stds(config, "paik", solver, solver, paik_solve)


# def nsf(config: Config_Diversity, solver: Solver):
#     nsf = Solver(solver_param=PANDA_NSF, load_date="0115-0234", work_dir=config.workdir)
#     iterate_over_base_stds(config, "nsf", nsf, solver, nsf_solve)


if __name__ == "__main__":
    config = Config_Diversity()
    # config.date = "2024_03_04"
    # klampt_numerical_ik_solver(config, solver)
    # paik(config, solver)
    # nsf(config, solver)
    config.num_poses = 100
    config.num_sols = 100

    nsf_solver = Solver(solver_param=PANDA_NSF, load_date="best", work_dir=config.workdir)
    _, P = nsf_solver.robot.sample_joint_angles_and_poses(n=config.num_poses)
    J_nsf = random_ikp(nsf_solver, P, config.num_sols, nsf_batch, std=0.001)

    # paik_solver = Solver(solver_param=PANDA_PAIK, load_date="best", work_dir=config.workdir)
    # J_paik = random_ikp(paik_solver, P, config.num_sols, paik_batch, std=0.001)
    # J_num = random_ikp(paik_solver, P, config.num_sols, numerical_inverse_kinematics_batch)
