import time
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.cluster import AgglomerativeClustering
from pafik.solver import Solver
from pafik.settings import DEFULT_SOLVER

import torch
from ikflow.utils import set_seed
from ikflow.model_loading import get_ik_solver
from jkinpylib.evaluation import solution_pose_errors

from tabulate import tabulate

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


def cluster_based_on_distance(a, dist_thresh=1):
    if len(a) < 3:
        # a minimum of 2 is required by AgglomerativeClustering.
        return 0
    kmeans = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_thresh
    ).fit(a)
    return kmeans.n_clusters_
    # return a[np.sort(np.unique(kmeans.labels_, return_index=True)[1])]


def n_solutions_to_reach_n_clusters(J_pose, l2_, ang_):
    """
    # example of useage
    # n_solutions_to_reach_n_clusters(J_hat[i], [10, 15], l2[i], ang[i], 2)
    """
    assert all(n_cluster > 2 for n_cluster in N_CLUSTERS_THRESHOLD)

    default_ = -1
    result_n_clusters = np.zeros((len(N_CLUSTERS_THRESHOLD)))
    record = np.full(len(J_pose), default_)
    bound = len(J_pose) - 1

    for i, n_cluster in enumerate(N_CLUSTERS_THRESHOLD):
        l, r = 0, bound
        # binary search
        while l < r:
            m = (l + r) // 2

            if record[m] == default_:
                J_partial, l2_partial, ang_partial = J_pose[:m], l2_[:m], ang_[:m]
                J_valid = J_partial[
                    (l2_partial < LAMBDA[0]) & (ang_partial < LAMBDA[1])
                ]
                record[m] = cluster_based_on_distance(
                    J_valid, JOINT_CONFIG_RADS_DISTANCE_THRESHOLD
                )
            # print(f"l: {l}, r: {r}, m: {m}, record[m]: {record[m]}")
            if record[m] < n_cluster:
                l = m + 1
            else:
                r = m

        # print(f"record: {record}")
        if r == bound:
            result_n_clusters[i] = np.nan
        else:
            result_n_clusters[i] = r
    return result_n_clusters


def n_cluster_analysis(J_hat_, l2_, ang_):
    assert len(J_hat_) == len(l2_) == len(ang_) == NUM_POSES
    return np.array(
        [
            n_solutions_to_reach_n_clusters(J_hat_[i], l2_[i], ang_[i])
            for i in trange(NUM_POSES)
        ]
    )


class Generate_Diverse_Postures_Info:
    def __init__(
        self,
        name,
        average_time_per_pose,
        num_poses,
        num_sols,
        df,
        success_rate_threshold,
    ):
        # check df counts at least .95 of num_sols
        assert (
            df.count().min() / num_poses > success_rate_threshold
        ), f"df counts at least {success_rate_threshold} of num_sols, {df.count().values / num_poses}"
        self.name = name
        self.success_rate = df.count().values / num_poses
        self.average_time_per_pose = average_time_per_pose
        self.average_time = average_time_per_pose / num_sols
        self.num_sols = num_sols
        self.df = df

        # def get_average_time_for_n_clusters(self):
        mean_of_num_solutions_to_reach_n_clusters = self.df.mean().values
        average_time_to_reach_n_clusters = (
            mean_of_num_solutions_to_reach_n_clusters * self.average_time
        )
        # return average_time_to_reach_n_clusters
        self.average_time_to_reach_n_clusters = average_time_to_reach_n_clusters

    def __repr__(self):
        return f"Average time per pose: {self.name}: {self.average_time:.6f} s ({self.average_time_per_pose:.2f} s / {self.num_sols} IK solutions)"

    def get_list_of_name_and_average_time_to_reach_n_clusters(self):
        return [self.name, *self.average_time_to_reach_n_clusters]

    def get_list_of_name_num_sols_and_average_time_to_reach_n_clusters(self):
        return [f"{self.name} ({self.num_sols})", *self.success_rate]


def main():
    solver_param = DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver = Solver(solver_param=solver_param)
    num_poses = NUM_POSES
    num_sols = NUM_SOLS
    n_neighbors = N_NEIGHBORS
    verbose = True
    batch_size = BATCH_SIZE
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
    solver.base_std = STD
    J_hat = solver.solve_batch(
        P_expand_dim, F, 1, batch_size=batch_size, verbose=verbose
    )
    l2, ang = solver.evaluate_pose_error_J3d_P2d(J_hat, P_expand_dim, return_all=True)
    average_time = (time.time() - begin_time) / num_poses
    J_hat = J_hat.reshape(NUM_POSES, N_NEIGHBORS, -1)
    l2 = l2.reshape(NUM_POSES, N_NEIGHBORS)
    ang = ang.reshape(NUM_POSES, N_NEIGHBORS)

    df = pd.DataFrame(n_cluster_analysis(J_hat, l2, ang), columns=N_CLUSTERS_THRESHOLD)
    print(df.info())
    print(df.describe())

    # IKFlow
    set_seed()
    # Build IKFlowSolver and set weights
    ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
    l2_flow = np.zeros((NUM_SOLS, len(P)))
    ang_flow = np.zeros((NUM_SOLS, len(P)))
    J_flow = torch.empty((NUM_SOLS, len(P), 7), dtype=torch.float32, device="cpu")

    begin_time = time.time()
    if num_poses < num_sols:
        for i in trange(num_poses):
            J_flow[:, i, :] = ik_solver.solve(
                P[i],
                n=num_sols,
                latent_scale=STD,
                refine_solutions=False,
                return_detailed=False,
            ).cpu()  # type: ignore

            l2_flow[:, i], ang_flow[:, i] = solution_pose_errors(
                ik_solver.robot, J_flow[:, i, :], P[i]
            )
    else:
        for i in trange(num_sols):
            J_flow[i] = ik_solver.solve_n_poses(
                P, latent_scale=STD, refine_solutions=False, return_detailed=False
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
        ),
        columns=N_CLUSTERS_THRESHOLD,
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
