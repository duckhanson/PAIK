from dataclasses import dataclass

@dataclass(frozen=True)
class CONFIG_IKP:
    # commons
    num_poses: int = 300
    num_sols: int = 100
    std: float = 0.25
    success_threshold: tuple = (5e-3, 2)
    
    # paik
    workdir: str = "/home/luca/paik"
    batch_size: int = 5000
    use_nsf_only = False
    method_of_select_reference_posture = "knn"
    
    # nodeik
    nodeik_workdir: str = "/home/luca/nodeik"
    urdf_path: str = f"{nodeik_workdir}/examples/assets/robots/franka_panda/panda_arm.urdf"    
    model_path: str = f"{nodeik_workdir}/model/panda_loss-20.ckpt"
    
    