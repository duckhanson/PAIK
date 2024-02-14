import numpy as np
from tqdm import trange
from sklearn.cluster import AgglomerativeClustering

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


def n_solutions_to_reach_n_clusters(J_pose, l2_, ang_, n_clusters_threshold, lambda_, joint_config_rads_distance_threshold):
    """
    # example of useage
    # n_solutions_to_reach_n_clusters(J_hat[i], [10, 15], l2[i], ang[i], 2)
    """
    assert all(n_cluster > 2 for n_cluster in n_clusters_threshold)

    default_ = -1
    result_n_clusters = np.zeros((len(n_clusters_threshold)))
    record = np.full(len(J_pose), default_)
    bound = len(J_pose) - 1
    
    # print(f"J_pose.shape: {J_pose.shape}, l2_.shape: {l2_.shape}, ang_.shape: {ang_.shape}, n_clusters_threshold: {n_clusters_threshold}, lambda_: {lambda_}, joint_config_rads_distance_threshold: {joint_config_rads_distance_threshold}")

    for i, n_cluster in enumerate(n_clusters_threshold):
        l, r = 0, bound
        # binary search
        while l < r:
            m = (l + r) // 2

            if record[m] == default_:
                J_partial, l2_partial, ang_partial = J_pose[:m], l2_[:m], ang_[:m]
                J_valid = J_partial[
                    (l2_partial < lambda_[0]) & (ang_partial < lambda_[1])
                ]
                record[m] = cluster_based_on_distance(
                    J_valid, joint_config_rads_distance_threshold   
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


def n_cluster_analysis(J_hat_, l2_, ang_, num_poses=NUM_POSES, n_clusters_threshold=N_CLUSTERS_THRESHOLD, lambda_=LAMBDA, joint_config_rads_distance_threshold=JOINT_CONFIG_RADS_DISTANCE_THRESHOLD):
    assert len(J_hat_) == len(l2_) == len(ang_) == num_poses, f'J_hat_.shape: {J_hat_.shape}, l2_.shape: {l2_.shape}, ang_.shape: {ang_.shape}, NUM_POSES: {NUM_POSES}'
    # print(f'J_hat_.shape: {J_hat_.shape}, l2_.shape: {l2_.shape}, ang_.shape: {ang_.shape}, NUM_POSES: {NUM_POSES}')
    return np.array(
        [
            n_solutions_to_reach_n_clusters(J_hat_[i], l2_[i], ang_[i], n_clusters_threshold, lambda_, joint_config_rads_distance_threshold)
            for i in trange(num_poses)
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
