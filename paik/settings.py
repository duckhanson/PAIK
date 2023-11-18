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
    random_perm: bool = True
    shrink_ratio: float = 0.61
    subnet_width: int = 1150
    num_transforms: int = 8

    # training
    opt_type: str = "adamw"
    lr: float = 3.6e-4
    lr_weight_decay: float = 1.8e-2
    decay_step_size: int = int(5e4)

    noise_esp: float = 1.9e-3
    noise_esp_decay: float = 0.92

    sche_type: str = "plateau"
    gamma: float = 8.4e-2

    batch_size: int = 128
    num_epochs: int = 15

    # inference
    ckpt_name: str = "1107-1013"
    enable_load_model: bool = True
    enable_normalize: bool = False
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
