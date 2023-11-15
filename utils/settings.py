import os
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import random
import numpy as np
import torch

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

ets_table = {
    "panda": 7,
    "al5d": 4,
    "fetch": 10,
    # 'fetchcamera': 5, #
    # 'frankie': 9, #
    # 'frankieomni': 10, #
    "lbr": 7,
    "mico": 4,
    "puma": 6,
    "ur10": 6,
    # 'valkyrie': 58, #
    # 'yumi': 14, #
    # means not suitable in our case.
}

@dataclass()
class NewConfig:
    # workdir
    current_folder_path, _ = os.path.split(
            os.path.realpath(__file__)
        )
    current_workdir_path, _ = os.path.split(
        os.path.realpath(current_folder_path)
    )
    workdir: str = current_workdir_path
    # robot
    robot_name: str = "panda"
    enable_normalize: bool = False
    # training
    N_train: int = 240_0000  # 2500_0000
    N_test: int = 5_0000  # 5_0000
    K: int = 100
    
    # data
    data_dir: str = f"{workdir}/data/{robot_name}"
    
    # train
    train_dir: str = f"{data_dir}/train"
    
    path_J_train: Callable[[int, int, int, str, int], str] = lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/J-{N_train}-{n}-{m}-{r}.npy"
    path_P_train: Callable[[int, int, int, str, int], str] = lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/P-{N_train}-{n}-{m}-{r}.npy"
    path_F: Callable[[int, int, int, str, int], str] = lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/F-{N_train}-{n}-{m}-{r}.npy"
    
    # val
    val_dir: str = f"{data_dir}/val"
    path_J_test: Callable[[int, int, int, str, int], str] = lambda n, m, r, val_dir=val_dir, N_test=N_test: f"{val_dir}/J-{N_test}-{n}-{m}-{r}.npy"
    path_P_test: Callable[[int, int, int, str, int], str] = lambda n, m, r, val_dir=val_dir, N_test=N_test: f"{val_dir}/P-{N_test}-{n}-{m}-{r}.npy"
    
    # hnne parameter
    weight_dir: str = f"{workdir}/weights/{robot_name}"
    max_num_data_hnne: int = 150_0000
    
    # knn parameter
    path_knn: Callable[[int, int, int, str, int, bool], str] = lambda n, m, r, train_dir=train_dir, N_train=N_train, enable_normalize=enable_normalize: f"{train_dir}/knn-{N_train}-{n}-{m}-{r}-norm{enable_normalize}.pickle"
    
    # experiment
    traj_dir: str = f"{data_dir}/trajectory/"
    
    dir_paths: Tuple[str, str, str] = (data_dir, weight_dir, traj_dir)
    
    


class Config:
    def __init__(self):
        # workdir
        current_folder_path, current_folder_name = os.path.split(
            os.path.realpath(__file__)
        )
        current_workdir_path, current_folder_name = os.path.split(
            os.path.realpath(current_folder_path)
        )
        self.workdir = current_workdir_path
        # robot
        self.robot_name = "panda"
        self.enable_normalize = False
        self.n = ets_table[self.robot_name]  # n = dof
        # self.m = 3 + 4 # position(x, y, z) + quaternion
        self.m = 3  # position(x, y, z)
        # self.r = self.n - self.m  # degrees of redundancy r = n - m
        # self.r = 1
        self.r = 4
        # training
        # self.N_train = 2500_0000 # 2500_0000
        self.N_train = 240_0000  # 2500_0000
        self.N_test = 5_0000  # 5_0000
        self.K = 100

        # data
        self.data_dir = f"{self.workdir}/data/{self.robot_name}/"

        # train
        self.train_dir = self.data_dir + "train/"
        self.path_J_train = (
            lambda n=self.n, m=self.m, r=self.r: self.train_dir
            + f"J-{self.N_train}-{n}-{m}-{r}.npy"
        )  # joint configuration
        self.path_P_train = (
            lambda n=self.n, m=self.m, r=self.r: self.train_dir
            + f"P-{self.N_train}-{n}-{m}-{r}.npy"
        )  # end-effector position
        # self.path_J_train = self.train_dir + f"J-{self.N_train}-{self.n}-{self.m}-{self.r}.npy"  # joint configuration
        # self.path_P_train = self.train_dir + f"P-{self.N_train}-{self.n}-{self.m}-{self.r}.npy"  # end-effector position
        self.path_F = (
            lambda n=self.n, m=self.m, r=self.r: self.train_dir
            + f"F-{self.N_train}-{n}-{m}-{r}.npy"
        )  # hnne reduced feature vector
        # self.path_F = (
        #     self.train_dir + f"F-{self.N_train}-{self.n}-{self.m}-{self.r}.npy"
        # )  # hnne reduced feature vector

        # val
        self.val_dir = self.data_dir + "val/"
        self.path_J_test = (
            lambda n=self.n, m=self.m, r=self.r: self.val_dir
            + f"J-{self.N_test}-{n}-{m}-{r}.npy"
        )  # joint configuration
        self.path_P_test = (
            lambda n=self.n, m=self.m, r=self.r: self.val_dir
            + f"P-{self.N_test}-{n}-{m}-{r}.npy"
        )  # end-effector position
        # self.path_J_test = self.val_dir + f"J-{self.N_test}-{self.n}-{self.m}-{self.r}.npy"  # joint configuration
        # self.path_P_test = self.val_dir + f"P-{self.N_test}-{self.n}-{self.m}-{self.r}.npy"  # end-effector position
        self.path_F_test = None  # hnne reduced feature vector

        # hnne parameter
        self.weight_dir = f"{self.workdir}/weights/{self.robot_name}/"
        # self.num_neighbors = 10000
        self.max_num_data_hnne = 150_0000

        # knn parameter
        self.path_knn = (
            lambda n=self.n, m=self.m, r=self.r: self.train_dir
            + f"knn-{self.N_train}-{n}-{m}-{r}-norm{self.enable_normalize}.pickle"
        )
        # self.path_knn = self.train_dir + f"knn-{self.N_train}-{self.n}-{self.m}-{self.r}-norm{self.enable_normalize}.pickle"

        # flow parameter
        self.use_pretrained = True
        self.architecture = "nsf"
        self.device = "cuda"
        self.num_conditions = (
            self.m + self.r + 1
        )  # position + posture + noise = 3-dim + 4-dim + 1-dim
        self.num_transforms = 7
        self.subnet_width = 1024
        self.subnet_num_layers = 3

        # sflow parameter
        self.shrink_ratio = 0.61

        self.lr = 4.8e-4
        self.lr_weight_decay = 5e-3
        self.decay_gamma = 0.5
        self.decay_step_size = int(4e4)
        self.batch_size = 128
        self.noise_esp = 1e-3
        self.num_epochs = 10
        self.num_steps_eval = 1000
        self.num_steps_save = 3_0000
        self.num_eval_size = 100
        self.path_solver = self.weight_dir + f"{self.architecture}.pth"

        # experiment
        self.num_steps_add_noise = 1000
        self.traj_dir = self.data_dir + "trajectory/"

        self.dir_paths = [
            self.data_dir,
            self.weight_dir,
            self.traj_dir,
        ]

    def __repr__(self):
        return str(self.__dict__)


# config = Config()

config = NewConfig()

if __name__ == "__main__":
    current_folder_path, current_folder_name = os.path.split(os.path.realpath(__file__))
    current_workdir_path, current_folder_name = os.path.split(
        os.path.realpath(current_folder_path)
    )
    print(current_workdir_path)
    # print(config)
