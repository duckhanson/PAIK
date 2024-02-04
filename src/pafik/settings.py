import os
from dataclasses import dataclass
from typing import Tuple


@dataclass()
class SolverConfig:
    # robot
    robot_name: str = "panda"
    n: int = 7
    m: int = 7
    r: int = 1

    # model
    subnet_num_layers: int = 3
    model_architecture: str = "nsf"
    randperm: bool = False
    base_std: float = 0.68
    subnet_width: int = 1024
    num_transforms: int = 8
    num_bins: int = 10

    # training
    lr: float = 3.7e-4
    lr_weight_decay: float = 1.2e-2
    lr_amsgrad: bool = False
    lr_beta: Tuple[float, float] = (0.9, 0.999)

    noise_esp: float = 2.5e-3
    noise_esp_decay: float = 0.97

    gamma: float = 8.6e-2

    batch_size: int = 2048
    num_epochs: int = 15
    shce_patience: int = 2
    posture_feature_scale: float = 1.0
    use_nsf_only: bool = False
    select_reference_posture_method: str = "knn"

    # inference
    ckpt_name: str = "1118-0317"
    enable_load_model: bool = True
    device: str = "cuda"

    # workdir
    __current_folder_path, _ = os.path.split(os.path.realpath(__file__))
    __current_workdir_path, _ = os.path.split(os.path.realpath(__current_folder_path))
    _workdir: str = __current_workdir_path

    # training
    # (N, max) = (1000_0000, 430_0000), (500_0000, 400_0000)
    N: int = 500_0000  # 2500_0000

    # data
    data_dir: str = f"{_workdir}/data/{robot_name}"

    # train
    train_dir: str = f"{data_dir}/train"

    # hnne parameter
    weight_dir: str = f"{_workdir}/weights/{robot_name}"
    max_num_data_hnne: int = 400_0000

    # experiment
    traj_dir: str = f"{data_dir}/trajectory/"

    dir_paths: Tuple[str, str, str] = (data_dir, weight_dir, traj_dir)

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, value: str):
        self._workdir = value
        self.data_dir = f"{value}/data/{self.robot_name}"
        self.train_dir = f"{self.data_dir}/train"
        self.weight_dir = f"{value}/weights/{self.robot_name}"
        self.traj_dir = f"{self.data_dir}/trajectory/"
        self.dir_paths = (self.data_dir, self.weight_dir, self.traj_dir)


# DEFAULT_SOLVER_PARAM_M7_NORM = SolverConfig(
#     lr=0.00037,
#     gamma=0.086,
#     noise_esp=0.0025,
#     randperm=False,
#     base_std=0.68,
#     subnet_width=1024,
#     num_transforms=8,
#     lr_weight_decay=0.012,
#     noise_esp_decay=0.97,
#     subnet_num_layers=3,
#     batch_size=1024,
#     ckpt_name="1202-1325",  # 1202-1325, 1205-0023
# )

DEFAULT_NSF = SolverConfig(
    lr=0.00037,
    gamma=0.086,
    noise_esp=0.0025,
    randperm=False,
    base_std=0.68,
    subnet_width=1024,
    num_transforms=8,
    lr_weight_decay=0.012,
    noise_esp_decay=0.97,
    subnet_num_layers=3,
    batch_size=1024,
    use_nsf_only=True,
    ckpt_name="0115-0234",
)

DEFULT_SOLVER = SolverConfig(
    lr=4e-4,
    gamma=0.086,
    noise_esp=0.0025,
    randperm=False,
    base_std=0.65,
    subnet_width=1024,
    num_transforms=8,
    lr_weight_decay=0.013,
    shce_patience=2,
    noise_esp_decay=0.97,
    subnet_num_layers=3,
    batch_size=1024,
    use_nsf_only=False,
    ckpt_name="0126-1535",  # "0119-1047", "0118-0827", "0126-1535"
)


if __name__ == "__main__":
    print(SolverConfig())
