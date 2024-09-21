import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime


@dataclass()
class Config_File:
    iksolver_names: List[str] = field(
        default_factory=lambda: ["IKFlow", "NODEIK", "NSF", "PAIK"]
    )
    batch_size: int = 5000

    # paik
    workdir: str = "/home/luca/paik"

    # nodeik
    _nodeik_workdir: str = "/home/luca/nodeik"
    nodeik_urdf_path: str = (
        f"{_nodeik_workdir}/examples/assets/robots/franka_panda/panda_arm.urdf"
    )
    nodeik_model_path: str = f"{_nodeik_workdir}/model/panda_loss-20.ckpt"

    _date: str = field(
        default_factory=lambda: datetime.today().strftime("%Y_%m_%d")
    )  # "2024_02_24"

    def __post_init__(self):
        self.update_record_dir()

    @property
    def date(self):
        return self._date

    @property
    def nodeik_workdir(self):
        return self._nodeik_workdir

    @date.setter
    def date(self, value):
        self._date = value
        self.update_record_dir()

    @nodeik_workdir.setter
    def nodeik_workdir(self, value):
        self._nodeik_workdir = value
        self.nodeik_urdf_path = f"{value}/examples/assets/robots/franka_panda/panda_arm.urdf"
        self.nodeik_model_path = f"{value}/model/panda_loss-20.ckpt"

    def update_record_dir(self):
        self.record_dir = f"{self.workdir}/record/{self.date}"
        os.makedirs(self.record_dir, exist_ok=True)


@dataclass()
class Config_IKP(Config_File):
    # commons
    num_poses: int = 500
    num_sols: int = 250
    std: float = 0.001
    # success_threshold: Tuple = (5e-3, 2)

    # paik
    # use_nsf_only: bool = False
    # select_reference: str = "knn"


@dataclass()
class Config_Diversity(Config_IKP):
    # commons
    base_stds: list = field(
        default_factory=lambda: list(np.arange(0.1, 1.5, 0.1)))

    # nodeik
    pose_error_threshold: Tuple = (3e-2, 30)  # l2 (m), ang (deg)


@dataclass()
class Config_Posture(Config_File):
    # commons
    num_poses: int = 3000
    num_sols: int = 2000
    std: float = 0.25
    use_nsf_only: bool = False

    success_distance_thresholds: List[int] = field(
        default_factory=lambda: list(range(20, 300, 20))
    )
