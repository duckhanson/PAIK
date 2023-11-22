import os
from dataclasses import dataclass
from typing import Tuple


@dataclass()
class SolverConfig:
    # robot
    robot_name: str = "panda"
    nmr: Tuple[int, int, int] = (7, 7, 1)

    # model
    subnet_num_layers: int = 3
    model_architecture: str = "nsf"
    random_perm: bool = False
    shrink_ratio: float = 0.68
    subnet_width: int = 1024
    num_transforms: int = 8

    # training
    opt_type: str = "adamw"
    lr: float = 3.7e-4
    lr_weight_decay: float = 1.2e-2
    decay_step_size: int = int(4e4)

    noise_esp: float = 2.5e-3
    noise_esp_decay: float = 0.97

    sche_type: str = "plateau"
    gamma: float = 8.6e-2

    batch_size: int = 128
    num_epochs: int = 15

    # inference
    ckpt_name: str = "1118-0317"
    enable_load_model: bool = True
    enable_normalize: bool = True
    device: str = "cuda"

    # workdir
    __current_folder_path, _ = os.path.split(os.path.realpath(__file__))
    __current_workdir_path, _ = os.path.split(os.path.realpath(__current_folder_path))
    workdir: str = __current_workdir_path

    # training
    N_train: int = 240_0000  # 2500_0000
    N_test: int = 5_0000  # 5_0000

    # data
    data_dir: str = f"{workdir}/data/{robot_name}"

    # train
    train_dir: str = f"{data_dir}/train"

    # val
    val_dir: str = f"{data_dir}/val"

    # hnne parameter
    weight_dir: str = f"{workdir}/weights/{robot_name}"
    max_num_data_hnne: int = 150_0000

    # experiment
    traj_dir: str = f"{data_dir}/trajectory/"

    dir_paths: Tuple[str, str, str] = (data_dir, weight_dir, traj_dir)


if __name__ == "__main__":
    print(SolverConfig())
