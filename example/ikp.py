# Import required packages
import time
import numpy as np
from tqdm import trange
import torch
from paik.solver import get_solver
from common.config import Config_IKP
from common.display import display_ikp, save_ikp


def paik():
    config = Config_IKP()
    solver = get_solver(arch_name="paik", robot_name="atlas_waist_arm",
                        load=True, work_dir=config.workdir)

    (l2, ang, avg_inference_time) = solver.random_ikp(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
        select_reference=config.select_reference,
    )  # type: ignore

    save_ikp(config.record_dir, "paik", l2, ang,
             avg_inference_time)  # type: ignore


def nsf(robot_name: str):
    config = Config_IKP()
    solver = get_solver(arch_name="nsf", robot_name=robot_name,
                        load=True, work_dir=config.workdir)

    (l2, ang, avg_inference_time) = solver.random_ikp(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
    )  # type: ignore

    save_ikp(config.record_dir, "nsf", l2, ang,
             avg_inference_time)  # type: ignore


if __name__ == "__main__":
    paik()
    nsf()
