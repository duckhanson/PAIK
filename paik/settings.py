import os
from dataclasses import dataclass
from typing import List, Tuple, Callable


@dataclass()
class SolverConfig:
    lr: float = 3.6e-4
    gamma: float = 8.4e-2
    opt_type: str = "adamw"
    noise_esp: float = 1.9e-3
    sche_type: str = "plateau"
    batch_size: int = 128
    num_epochs: int = 15
    random_perm: bool = True
    subnet_width: int = 1150
    num_transforms: int = 8
    decay_step_size: int = int(5e4)
    lr_weight_decay: float = 1.8e-2
    noise_esp_decay: float = 0.92
    subnet_num_layers: int = 3
    model_architecture: str = "nsf"
    shrink_ratio: float = 0.61
    ckpt_name: str = "1107-1013"
    nmr: Tuple[int, int, int] = (7, 7, 1)
    enable_load_model: bool = True
    device: str = "cuda"


@dataclass()
class Developer:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


@dataclass()
class DataClassConfig:
    # workdir
    __current_folder_path, _ = os.path.split(os.path.realpath(__file__))
    __current_workdir_path, _ = os.path.split(os.path.realpath(__current_folder_path))
    workdir: str = __current_workdir_path
    # robot
    robot_name: str = "panda"
    enable_normalize: bool = False
    # training
    N_train: int = 240_0000  # 2500_0000
    N_test: int = 5_0000  # 5_0000

    # data
    data_dir: str = f"{workdir}/data/{robot_name}"

    # train
    train_dir: str = f"{data_dir}/train"

    path_J_train: Callable[
        [int, int, int, str, int], str
    ] = (
        lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/J-{N_train}-{n}-{m}-{r}.npy"
    )
    path_P_train: Callable[
        [int, int, int, str, int], str
    ] = (
        lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/P-{N_train}-{n}-{m}-{r}.npy"
    )
    path_F: Callable[
        [int, int, int, str, int], str
    ] = (
        lambda n, m, r, train_dir=train_dir, N_train=N_train: f"{train_dir}/F-{N_train}-{n}-{m}-{r}.npy"
    )

    # val
    val_dir: str = f"{data_dir}/val"
    path_J_test: Callable[
        [int, int, int, str, int], str
    ] = (
        lambda n, m, r, val_dir=val_dir, N_test=N_test: f"{val_dir}/J-{N_test}-{n}-{m}-{r}.npy"
    )
    path_P_test: Callable[
        [int, int, int, str, int], str
    ] = (
        lambda n, m, r, val_dir=val_dir, N_test=N_test: f"{val_dir}/P-{N_test}-{n}-{m}-{r}.npy"
    )

    # hnne parameter
    weight_dir: str = f"{workdir}/weights/{robot_name}"
    max_num_data_hnne: int = 150_0000

    # knn parameter
    path_knn: Callable[
        [int, int, int, str, int, bool], str
    ] = (
        lambda n, m, r, train_dir=train_dir, N_train=N_train, enable_normalize=enable_normalize: f"{train_dir}/knn-{N_train}-{n}-{m}-{r}-norm{enable_normalize}.pickle"
    )

    # experiment
    traj_dir: str = f"{data_dir}/trajectory/"

    dir_paths: Tuple[str, str, str] = (data_dir, weight_dir, traj_dir)


config = DataClassConfig()

if __name__ == "__main__":
    print(config)
