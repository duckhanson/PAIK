import os
import random

import numpy as np
import torch
from torch.nn import LeakyReLU

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
        self.n = ets_table[self.robot_name]  # n = dof
        self.m = 3  # position(x, y, z)
        self.r = self.n - self.m  # degrees of redundancy r = n - m

        # data
        self.data_dir = f"{self.workdir}/data/{self.robot_name}/"

        # train
        self.train_dir = self.data_dir + "train/"
        self.path_J_train = self.train_dir + "feature.npy"  # joint configuration
        self.path_P_train = self.train_dir + "target.npy"  # end-effector position
        self.path_F = (
            self.train_dir + "feature_trans.npy"
        )  # hnne reduced feature vector

        # val
        self.val_dir = self.data_dir + "val/"
        self.path_J_test = self.val_dir + "feature.npy"  # joint configuration
        self.path_J_test = self.val_dir + "target.npy"  # end-effector position
        self.path_F_test = (
            self.val_dir + "feature_trans.npy"
        )  # hnne reduced feature vector

        # hnne parameter
        self.weight_dir = f"{self.workdir}/weights/{self.robot_name}/"
        self.num_neighbors = 1000
        self.path_hnne = self.weight_dir + "hnne.pickle"

        # knn parameter
        self.path_knn = self.weight_dir + "knn.pickle"

        # flow parameter
        self.use_pretrained = False
        self.architecture = "nsf"
        self.device = "cuda"
        self.num_conditions = (
            self.n + 1
        )  # position + posture + noise = 3-dim + 4-dim + 1-dim
        self.num_transforms = 7
        self.subnet_width = 1024
        self.subnet_num_layers = 3
        self.activation = LeakyReLU

        # sflow parameter
        self.shrink_ratio = 0.61

        # training
        self.N_train = 250_0000
        self.N_test = 2_0000
        self.K = 500

        self.lr = 5e-7
        self.lr_weight_decay = 7e-3
        self.decay_gamma = 0.79
        self.decay_step_size = 30000
        self.batch_size = 128
        self.noise_esp = 1e-3
        self.num_epochs = 25
        self.num_steps_eval = 300
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


config = Config()

if __name__ == "__main__":
    current_folder_path, current_folder_name = os.path.split(os.path.realpath(__file__))
    current_workdir_path, current_folder_name = os.path.split(
        os.path.realpath(current_folder_path)
    )
    print(current_workdir_path)
    # print(config)
