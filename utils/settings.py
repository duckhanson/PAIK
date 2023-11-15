import os
from dataclasses import dataclass
from typing import List, Tuple, Callable


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
