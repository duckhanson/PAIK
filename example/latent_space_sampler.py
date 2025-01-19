from typing import Optional

import numpy as np
import torch
from sklearn.cluster import BisectingKMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.contrib import itertools as tqdm_itertools
from common.evaluate import evaluate_pose_error_J3d_P2d
from paik.model import NSF

# set the same random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)


class Retriever:
    def __init__(self, nsf):
        self.nsf = nsf
        self.robot = self.nsf.robot
        self.P_all = nsf.P
        self.J_all = nsf.J
        self.Z_all = nsf.Z

        self.temp_cluster_info = {}

    def init(self, max_samples_list, n_clusters_list):
        print("Start to initialize cluster info...")
        for max_samples, n_clusters in tqdm_itertools.product(max_samples_list, n_clusters_list):
            self.get_cluster_info(max_samples, n_clusters)

    def get_cluster_info(self, max_samples: int, n_clusters: int):
        if f"{max_samples}_{n_clusters}" in self.temp_cluster_info:
            return self.temp_cluster_info[f"{max_samples}_{n_clusters}"]['centroids_ids']

        Z_samples = self.Z_all[:max_samples]
        # buliding the clustering
        cluster = BisectingKMeans(
            n_clusters=n_clusters, random_state=0).fit(Z_samples)
        centroids = cluster.cluster_centers_
        # find centroids ids in Z_samples
        Z_samples_knn = NearestNeighbors(n_neighbors=1).fit(Z_samples)
        centroids_ids = Z_samples_knn.kneighbors(
            centroids, return_distance=False).flatten()
        self.temp_cluster_info[f"{max_samples}_{n_clusters}"] = {
            'centroids_ids': centroids_ids
        }
        return centroids_ids

    def cluster_retriever(self, seeds: Optional[np.ndarray] = None, num_poses: int = 1, num_sols: int = 1, max_samples: int = 50000, radius: float = 0, n_clusters: int = 100):
        print(f"Start to cluster retriever with max_samples: {max_samples}, num_poses: {num_poses}, num_sols: {num_sols}, radius: {radius}, n_clusters: {n_clusters}")
        
        Z_samples = self.Z_all[:max_samples]
        J_samples = self.J_all[:max_samples]

        centroids_ids = self.get_cluster_info(max_samples, n_clusters)
        num_total = num_sols * num_poses

        # diversity
        if seeds is None:
            ids = np.random.choice(centroids_ids, num_total, replace=True)
        # selection
        else:
            assert seeds.ndim == 2, f"Invalid shape of seeds: {seeds.shape}"
            assert seeds.shape[0] == num_poses, f"Invalid shape of seeds: {seeds.shape}"

            # shape: (num_centroids, nsf.n)
            J_centroid = J_samples[centroids_ids]
            # k (3%) nearest neighbors of centroids
            k = np.floor(len(centroids_ids) / 30).astype(int)
            k = max(1, k)
            k = 1
            J_knn = NearestNeighbors(n_neighbors=k).fit(J_centroid)

            ids = np.empty((num_sols, num_poses), dtype=int)
            for i in range(len(seeds)):
                # id shape: (1, k)
                id = J_knn.kneighbors(seeds[i].reshape(1, -1), return_distance=False)
                # ids[:, i] shape: (num_sols), random select num_sols from k
                ids[:, i] = np.random.choice(id.flatten(), num_sols, replace=True)
            # shape: (num_sols, num_poses)  -> shape: (num_sols * num_poses)
            ids = ids.flatten()
        Z_out = Z_samples[ids]
        noise = np.random.normal(0, radius, size=Z_out.shape)
        return Z_out + noise

    def random_retriever(self, seeds: Optional[np.ndarray] = None, num_poses: int = 1, num_sols: int = 1, max_samples: int = 1000, radius: float = 0):
        print(f"Start to random retriever with max_samples: {max_samples}, num_poses: {num_poses}, num_sols: {num_sols}, radius: {radius}")
        # diversity
        if seeds is None:
            ids = np.random.choice(
                max_samples, num_sols * num_poses, replace=True)
        # selection
        else:
            # k (3%) nearest neighbors of centroids
            k = np.floor(max_samples / 30).astype(int)
            k = max(1, k)
            k = 1
            J_knn = NearestNeighbors(n_neighbors=k).fit(
                self.J_all[:max_samples])
            ids = np.empty((num_sols, num_poses), dtype=int)
            for i in range(len(seeds)):
                # id shape: (1, k)
                id = J_knn.kneighbors(seeds[i].reshape(1, -1), return_distance=False)
                # ids[:, i] shape: (num_sols), random select num_sols from k
                ids[:, i] = np.random.choice(id.flatten(), num_sols, replace=True)
            # shape: (num_sols, num_poses)  -> shape: (num_sols * num_poses)
            ids = ids.flatten()
        Z_out = self.Z_all[ids]
        noise = np.random.normal(0, radius, size=Z_out.shape)
        return Z_out + noise

    def numerical_retriever(self, poses: np.ndarray, seeds: Optional[np.ndarray] = None, num_sols: int = 1, num_seeds_per_pose: int = 2, radius: float = 0):
        print("Start to numerical retriever...")

        assert poses.ndim == 2, f"Invalid shape of poses: {poses.shape}"
        num_poses = poses.shape[0]

        if seeds is None:
            ik_sols = self.get_diverse_ik_seeds(poses, num_sols, num_seeds_per_pose)
            if num_sols > 1:
                poses = np.expand_dims(poses, axis=0).repeat(
                    num_sols, axis=0).reshape(-1, poses.shape[-1])
                num_poses *= num_sols
        elif poses.ndim != seeds.ndim:
            raise ValueError(
                f"Invalid shape of poses: {poses.shape}, seeds: {seeds.shape}")
        else:
            ik_sols = np.empty_like(seeds)
            for i in range(num_poses):
                result = None
                seed = seeds[i]
                while result is None:
                    result = self.robot.inverse_kinematics_klampt(poses[i], seed=seed)
                    seed = None
                ik_sols[i] = result

            if num_sols > 1:
                poses = np.expand_dims(poses, axis=0).repeat(
                    num_sols, axis=0).reshape(-1, poses.shape[-1])
                ik_sols = np.expand_dims(ik_sols, axis=0).repeat(
                    num_sols, axis=0).reshape(-1, ik_sols.shape[-1])
                num_poses *= num_sols

        Z_out = self.nsf.generate_z_from_ik_solutions(poses, ik_sols)
        noise = np.random.normal(0, radius, size=Z_out.shape)
        return Z_out + noise

    def get_diverse_ik_seeds(self, poses: np.ndarray, num_sols: int = 1, num_seeds_per_pose: int = 2):
        print(f"Start to get diverse ik seeds w/ num_sols: {num_sols}, num_seeds_per_pose: {num_seeds_per_pose}")
        assert poses.ndim == 2, f"Invalid shape of poses: {poses.shape}"
        num_poses = poses.shape[0]
        num_seeds = num_seeds_per_pose * num_poses         
        seeds = self.robot.sample_joint_angles(num_seeds)
        seeds = seeds.reshape(num_seeds_per_pose, num_poses, seeds.shape[-1])

        ik_sols = np.empty_like(seeds)
        for i in range(num_seeds_per_pose):
            for j in range(num_poses):
                result = None
                seed = seeds[i, j]
                while result is None:
                    result = self.robot.inverse_kinematics_klampt(poses[j], seed=seed)
                    seed = None
                ik_sols[i, j] = result
        
        if num_sols > 1:
            poses = np.expand_dims(poses, axis=0).repeat(
                num_sols, axis=0).reshape(-1, poses.shape[-1])
            scale = np.ceil(num_sols / num_seeds_per_pose).astype(int)
            ik_sols = ik_sols.repeat(scale, axis=0)
            random_choice = np.random.choice(num_seeds_per_pose, num_sols, replace=True)
            ik_sols = ik_sols[random_choice].reshape(-1, ik_sols.shape[-1])
            num_poses *= num_sols

        return ik_sols
