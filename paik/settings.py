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
    lr: float = 3.7e-4
    lr_weight_decay: float = 1.2e-2

    noise_esp: float = 2.5e-3
    noise_esp_decay: float = 0.97

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


DEFAULT_SOLVER_PARAM_M3 = SolverConfig(
    lr=0.00033,
    gamma=0.094,
    noise_esp=0.001,
    batch_size=128,
    num_epochs=10,
    random_perm=True,
    subnet_width=1024,
    num_transforms=10,
    lr_weight_decay=0.013,
    noise_esp_decay=0.8,
    subnet_num_layers=3,
    model_architecture="nsf",
    shrink_ratio=0.61,
    ckpt_name="0930-0346",
    nmr=(7, 3, 4),
    enable_load_model=True,
    device="cuda",
)

DEFAULT_SOLVER_PARAM_M7 = SolverConfig(
    lr=0.00036,
    gamma=0.084,
    noise_esp=0.0019,
    batch_size=128,
    num_epochs=15,
    random_perm=True,
    subnet_width=1150,
    num_transforms=8,
    lr_weight_decay=0.018,
    noise_esp_decay=0.92,
    subnet_num_layers=3,
    model_architecture="nsf",
    shrink_ratio=0.61,
    ckpt_name="1107-1013",
    nmr=(7, 7, 1),
    enable_load_model=True,
    device="cuda",
)

DEFAULT_SOLVER_PARAM_M7_NORM = SolverConfig(
    lr=0.00037,
    gamma=0.086,
    noise_esp=0.0025,
    random_perm=False,
    shrink_ratio=0.68,
    subnet_width=1024,
    num_transforms=8,
    lr_weight_decay=0.012,
    noise_esp_decay=0.97,
    enable_normalize=True,
    subnet_num_layers=3,
    ckpt_name="1129-0817",  # "1128-0459", "1128-0857", "1129-0817"
)


if __name__ == "__main__":
    print(SolverConfig())
