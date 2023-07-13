import os

import torch
import random
import numpy as np
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
        self.dof = ets_table[self.robot_name]

        # data
        self.data_dir = f"{self.workdir}/data/{self.robot_name}/"

        # train
        self.train_dir = self.data_dir + "train/"
        self.x_train_path = self.train_dir + "feature.npy"  # joint configuration
        self.y_train_path = self.train_dir + "target.npy"  # end-effector position
        self.x_trans_train_path = (
            self.train_dir + "feature_trans.npy"
        )  # hnne reduced feature vector

        # val
        self.val_dir = self.data_dir + "val/"
        self.x_val_path = self.val_dir + "feature.npy"  # joint configuration
        self.y_val_path = self.val_dir + "target.npy"  # end-effector position
        self.x_trans_val_path = (
            self.val_dir + "feature_trans.npy"
        )  # hnne reduced feature vector

        # hnne parameter
        self.weight_dir = f"{self.workdir}/weights/{self.robot_name}/"
        self.reduced_dim = self.dof - 3
        self.num_neighbors = 1000
        self.hnne_save_path = self.weight_dir + "hnne.pickle"

        # flow parameter
        self.use_pretrained = False
        self.architecture = "nsf"
        self.device = "cuda"
        self.num_features = self.dof
        self.num_conditions = (
            3 + self.reduced_dim + 1
        )  # position + posture + noise = 3-dim + 4-dim + 1-dim
        self.num_transforms = 7
        self.subnet_shape = [1024] * 3
        self.activation = LeakyReLU

        # sflow parameter
        self.shrink_ratio = 0.61

        # training
        self.num_train_size = 250_0000
        self.num_val_size = 2_0000
        self.lr = 4.668e-7
        self.lr_weight_decay = 7e-3
        self.decay_gamma = 0.79
        self.decay_step_size = 30000
        self.batch_size = 128
        self.noise_esp = 1e-3
        self.num_epochs = 25
        self.num_steps_eval = 300
        self.num_steps_save = 3_0000
        self.num_eval_size = 100
        self.num_eval_samples = 500
        self.save_path = self.weight_dir + f"{self.architecture}.pth"

        # experiment
        self.num_steps_add_noise = 1000
        self.show_pose_dir = self.data_dir + "show_pose/"
        self.show_pose_features_path = self.show_pose_dir + "features.npy"
        self.show_pose_pidxs_path = self.show_pose_dir + "pidxs.npy"
        self.show_pose_errs_path = self.show_pose_dir + "errs.npy"
        self.show_pose_log_probs_path = self.show_pose_dir + "log_probs.npy"
        self.traj_dir = self.data_dir + "trajectory/"

        self.dir_paths = [
            self.data_dir,
            self.weight_dir,
            self.traj_dir,
            self.show_pose_dir,
        ]

    def __repr__(self):
        return str(self.__dict__)


config = Config()

if __name__ == "__main__":
    current_folder_path, current_folder_name = os.path.split(
        os.path.realpath(__file__))
    current_workdir_path, current_folder_name = os.path.split(
        os.path.realpath(current_folder_path)
    )
    print(current_workdir_path)
    # print(config)
