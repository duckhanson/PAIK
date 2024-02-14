import time
import numpy as np
import pandas as pd
from tqdm import trange
from pafik.solver import Solver
from pafik.settings import DEFULT_SOLVER

import torch
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from tabulate import tabulate
from evaluation import n_cluster_analysis, Generate_Diverse_Postures_Info

WORKDIR = "."
NUM_POSES = 5_000
N_NEIGHBORS = 5_000  # PAFIK
NUM_SOLS = 15_000  # IKFlow
BATCH_SIZE = 5_000
LAMBDA = (0.005, 0.05)
STD = 0.25
JOINT_CONFIG_RADS_DISTANCE_THRESHOLD = 2
N_CLUSTERS_THRESHOLD = [10, 15, 20, 25, 30]
SOLUTIONS_SUCCESS_RATE_THRESHOLD_FOR_CLUSTERING_IN_NUM_SOLS = 0.80


def main():
    solver_param = DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver = Solver(solver_param=solver_param)
    num_poses = NUM_POSES
    num_sols = NUM_SOLS
    n_neighbors = N_NEIGHBORS
    verbose = True
    batch_size = BATCH_SIZE
    lambda_ = LAMBDA
    std = STD
    joint_config_rads_distance_threshold = JOINT_CONFIG_RADS_DISTANCE_THRESHOLD
    n_clusters_threshold = N_CLUSTERS_THRESHOLD
    success_rate_thresold = SOLUTIONS_SUCCESS_RATE_THRESHOLD_FOR_CLUSTERING_IN_NUM_SOLS

    # PAFIK
    # P = (NUM_POSES, m)
    _, P = solver.robot.sample_joint_angles_and_poses(n=num_poses, return_torch=False)

    # F = (NUM_POSES * N_NEIGHBORS, r)
    F = solver.F[
        solver.P_knn.kneighbors(
            np.atleast_2d(P), n_neighbors=n_neighbors, return_distance=False
        ).flatten()
    ]

    P_expand_dim = np.repeat(np.expand_dims(P, axis=1), n_neighbors, axis=1).reshape(
        -1, P.shape[-1]
    )

    begin_time = time.time()
    solver.base_std = std
    J_hat = solver.solve_batch(
        P_expand_dim, F, 1, batch_size=batch_size, verbose=verbose
    )
    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P_expand_dim, return_all=True)
    average_time = (time.time() - begin_time) / num_poses
    J_hat = J_hat.reshape(num_poses, n_neighbors, -1)
    l2 = l2.reshape(num_poses, n_neighbors)
    ang = ang.reshape(num_poses, n_neighbors)

    df = pd.DataFrame(
        n_cluster_analysis(
            J_hat,
            l2,
            ang,
            num_poses,
            n_clusters_threshold=n_clusters_threshold,
            lambda_=lambda_,
            joint_config_rads_distance_threshold=joint_config_rads_distance_threshold,
        ),
        columns=N_CLUSTERS_THRESHOLD,
    )
    print(df.info())
    print(df.describe())

    # IKFlow
    set_seed()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    l2_flow = np.zeros((num_sols, len(P)))
    ang_flow = np.zeros((num_sols, len(P)))
    J_flow = torch.empty((num_sols, len(P), 7), dtype=torch.float32, device="cpu")

    begin_time = time.time()
    if num_poses < num_sols:
        for i in trange(num_poses):
            J_flow[:, i, :] = ik_solver.solve(
                P[i],
                n=num_sols,
                latent_scale=std,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2_flow[:, i], ang_flow[:, i] = solution_pose_errors(
                ik_solver.robot, J_flow[:, i, :], P[i]
            )
    else:
        for i in trange(num_sols):
            J_flow[i] = ik_solver.solve_n_poses(
                P, latent_scale=std, refine_solutions=False, return_detailed=False
            ).cpu()
            l2_flow[i], ang_flow[i] = solution_pose_errors(
                ik_solver.robot, J_flow[i], P
            )
    average_time_flow = (time.time() - begin_time) / num_poses

    df_flow = pd.DataFrame(
        n_cluster_analysis(
            J_flow.numpy().transpose((1, 0, 2)),
            l2_flow.transpose((1, 0)),
            ang_flow.transpose((1, 0)),
            num_poses,
            n_clusters_threshold=n_clusters_threshold,
            lambda_=lambda_,
            joint_config_rads_distance_threshold=joint_config_rads_distance_threshold,
        ),
        columns=n_clusters_threshold,
    )
    print(df_flow.info())
    print(df_flow.describe())

    # print a tbulate of the average time to reach n clusters
    P_posture_info = Generate_Diverse_Postures_Info(
        "P", average_time, num_poses, n_neighbors, df, success_rate_thresold
    )
    F_posture_info = Generate_Diverse_Postures_Info(
        "F", average_time_flow, num_poses, num_sols, df_flow, success_rate_thresold
    )
    ratio_average_time_to_reach_n_clusters = (
        P_posture_info.average_time_to_reach_n_clusters
        / F_posture_info.average_time_to_reach_n_clusters
    )
    name_append_ratio = ["ratio", *ratio_average_time_to_reach_n_clusters]

    print(
        tabulate(
            [
                N_CLUSTERS_THRESHOLD,
                P_posture_info.get_list_of_name_and_average_time_to_reach_n_clusters(),
                F_posture_info.get_list_of_name_and_average_time_to_reach_n_clusters(),
                name_append_ratio,
            ],
            headers="firstrow",
        )
    )

    # print a tbulate of the success rate of the clustering
    print(
        tabulate(
            [
                N_CLUSTERS_THRESHOLD,
                P_posture_info.get_list_of_name_num_sols_and_average_time_to_reach_n_clusters(),
                F_posture_info.get_list_of_name_num_sols_and_average_time_to_reach_n_clusters(),
            ],
            headers="firstrow",
        )
    )


if __name__ == "__main__":
    main()
