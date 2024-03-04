import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime


@dataclass()
class ConfigFile:
    iksolver_names: List[str] = field(
        default_factory=lambda: ["IKFlow", "PAIK", "NODEIK"]
    )

    # paik
    workdir: str = "/home/luca/paik"

    # nodeik
    nodeik_workdir: str = "/home/luca/nodeik"
    nodeik_urdf_path: str = (
        f"{nodeik_workdir}/examples/assets/robots/franka_panda/panda_arm.urdf"
    )
    nodeik_model_path: str = f"{nodeik_workdir}/model/panda_loss-20.ckpt"

    _date: str = field(
        default_factory=lambda: datetime.today().strftime("%Y_%m_%d")
    )  # "2024_02_24"

    def __post_init__(self):
        self.update_record_dir()

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = value
        self.update_record_dir()

    def update_record_dir(self):
        self.record_dir = f"{self.workdir}/record/{self.date}"
        os.makedirs(self.record_dir, exist_ok=True)


@dataclass()
class ConfigIKP(ConfigFile):
    # commons
    num_poses: int = 300
    num_sols: int = 100
    std: float = 0.25
    success_threshold: Tuple = (5e-3, 2)

    # paik
    batch_size: int = 5000
    use_nsf_only: bool = False
    method_of_select_reference_posture: str = "knn"


@dataclass()
class ConfigDiversity(ConfigFile):
    # commons
    num_poses: int = 2500
    num_sols: int = 1000
    base_stds: list = field(default_factory=lambda: list(np.arange(0.1, 1.5, 1)))

    # nodeik
    pose_error_threshold: Tuple = (3e-2, 30)  # l2 (m), ang (deg)


@dataclass()
class ConfigPosture(ConfigFile):
    # commons
    num_poses: int = 3000
    num_sols: int = 2000
    batch_size: int = 5000
    std: float = 0.25
    use_nsf_only: bool = False

    success_distance_thresholds: List[int] = field(
        default_factory=lambda: list(range(20, 300, 20))
    )
