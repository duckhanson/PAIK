import os
from dataclasses import dataclass
from typing import Tuple


@dataclass()
class SolverConfig:
    # robot
    robot_name: str = "panda"
    n: int = 7
    m: int = 7  # quaternion = [px, py, pz, qw, qx, qy, qz] the same as ikflow
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

    use_dimension_reduction: bool = False

    gamma: float = 8.6e-2

    batch_size: int = 2048
    num_epochs: int = 15
    shce_patience: int = 2
    use_nsf_only: bool = False
    get_reference_partition_label_method: str = "knn"

    # inference
    ckpt_name: str = "1118-0317"
    enable_load_model: bool = True
    device: str = "cuda"

    # workdir
    __current_folder_path, _ = os.path.split(os.path.realpath(__file__))
    _workdir, _ = os.path.split(os.path.realpath(__current_folder_path))

    # training
    # (N, max) = (1000_0000, 430_0000), (500_0000, 400_0000)
    N: int = 500_0000  # 2500_0000

    # data
    data_dir: str = f"{_workdir}/data/{robot_name}"

    # train
    train_dir: str = f"{data_dir}/train"

    # hnne parameter
    weight_dir: str = f"{_workdir}/weights/{robot_name}"
    max_num_data_hnne: int = 300_0000

    # experiment
    traj_dir: str = f"{data_dir}/trajectory/"

    dir_paths: Tuple[str, str, str, str] = (
        data_dir, weight_dir, traj_dir, train_dir)

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
        self.dir_paths = (self.data_dir, self.weight_dir,
                          self.traj_dir, self.train_dir)


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

PANDA_NSF = SolverConfig(
    subnet_width=1024,
    num_transforms=8,
    subnet_num_layers=3,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
    ckpt_name="0115-0234",
)

PANDA_PAIK = SolverConfig(
    subnet_width=1024,
    num_transforms=8,
    subnet_num_layers=3,
    ckpt_name="0126-1535",  # "0119-1047", "0118-0827", "0126-1535"
)

FETCH_PAIK = SolverConfig(
    robot_name="fetch",
    n=8,
    m=7,
    r=1,
    num_bins=6,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    ckpt_name="0621-0313",  # "0620-1327", "0620-0225"
)

FETCH_NSF = SolverConfig(
    robot_name="fetch",
    n=8,
    m=7,
    r=1,
    num_bins=6,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
)

FETCH_ARM_PAIK = SolverConfig(
    robot_name="fetch_arm",
    n=7,
    m=7,
    r=1,
    num_bins=6,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    ckpt_name="0622-0958",  # "0622-0958", "0621-2325"
)

FETCH_ARM_NSF = SolverConfig(
    robot_name="fetch_arm",
    n=7,
    m=7,
    r=1,
    num_bins=6,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
)

IIWA7_PAIK = SolverConfig(
    robot_name="iiwa7",
    n=7,
    m=7,
    r=1,
    num_bins=7,
    max_num_data_hnne=300_0000,
    num_transforms=8,
    ckpt_name="0624-1327",  # "0623-1803", "0624-1327"
)

ATLAS_ARM_PAIK = SolverConfig(
    robot_name="atlas_arm",
    n=6,
    m=7,
    r=1,
    max_num_data_hnne=300_0000,
    num_transforms=7,
)

ATLAS_ARM_NSF = SolverConfig(
    robot_name="atlas_arm",
    n=6,
    m=7,
    r=1,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
)

ATLAS_WAIST_ARM_PAIK = SolverConfig(
    robot_name="atlas_waist_arm",
    n=9,
    m=7,
    r=1,
    max_num_data_hnne=200_0000,
    num_transforms=7,
)

ATLAS_WAIST_ARM_NSF = SolverConfig(
    robot_name="atlas_waist_arm",
    n=9,
    m=7,
    r=1,
    max_num_data_hnne=200_0000,
    num_transforms=7,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
)

BAXTER_ARM_PAIK = SolverConfig(
    robot_name="baxter_arm",
    n=7,
    m=7,
    r=1,
    max_num_data_hnne=300_0000,
    num_transforms=7,
)

BAXTER_ARM_NSF = SolverConfig(
    robot_name="baxter_arm",
    n=7,
    m=7,
    r=1,
    max_num_data_hnne=300_0000,
    num_transforms=7,
    use_nsf_only=True,
    get_reference_partition_label_method="zero",
)

PR2_PAIK = SolverConfig(
    robot_name="pr2",
    n=8,
    m=7,
    r=1,
    max_num_data_hnne=300_0000,
    num_transforms=7,
)


def get_config(arch_name: str, robot_name: str):
    support_archs = ["paik", "nsf"]
    support_robots = ["panda", "fetch", "fetch_arm",
                      "atlas_arm", "atlas_waist_arm", "baxter_arm"]
    assert arch_name in support_archs, f"arch_name should be one of {support_archs}."
    assert robot_name in support_robots, f"robot_name should be one of {support_robots}."

    if robot_name == "panda":
        if arch_name == "nsf":
            return PANDA_NSF
        elif arch_name == "paik":
            return PANDA_PAIK
    elif robot_name == "fetch":
        if arch_name == "nsf":
            return FETCH_NSF
        elif arch_name == "paik":
            return FETCH_PAIK
    elif robot_name == "fetch_arm":
        if arch_name == "nsf":
            return FETCH_ARM_NSF
        elif arch_name == "paik":
            return FETCH_ARM_PAIK
    elif robot_name == "atlas_arm":
        if arch_name == "nsf":
            return ATLAS_ARM_NSF
        elif arch_name == "paik":
            return ATLAS_ARM_PAIK
    elif robot_name == "atlas_waist_arm":
        if arch_name == "nsf":
            return ATLAS_WAIST_ARM_NSF
        elif arch_name == "paik":
            return ATLAS_WAIST_ARM_PAIK
    elif robot_name == "baxter_arm":
        if arch_name == "nsf":
            return BAXTER_ARM_NSF
        elif arch_name == "paik":
            return BAXTER_ARM_PAIK
    else:
        raise ValueError(f"robot_name: {robot_name}, arch_name: {arch_name}")


if __name__ == "__main__":
    print(SolverConfig())
