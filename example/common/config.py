import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime


@dataclass()
class ConfigFile:
    # paik
    workdir: str = "/home/luca/paik"

    # nodeik
    nodeik_workdir: str = "/home/luca/nodeik"
    urdf_path: str = f"{nodeik_workdir}/examples/assets/robots/franka_panda/panda_arm.urdf"
    model_path: str = f"{nodeik_workdir}/model/panda_loss-20.ckpt"


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
    num_poses: int = 100
    num_sols: int = 200
    

    base_stds: list = field(default_factory=lambda: list(np.arange(0.1, 1.5, 1)))
    date: str = field(default_factory=lambda: datetime.today().strftime("%Y_%m_%d")) # "2024_02_24"
    
    # nodeik
    pose_error_threshold: Tuple = (3e-2, 30) # l2 (m), ang (deg)
    
    def __post_init__(self):
        self.record_dir = f"{self.workdir}/record/{self.date}"
        os.makedirs(self.record_dir, exist_ok=True)
        