# Import required packages
import time
import numpy as np
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import PANDA_NSF, PANDA_PAIK

from common.config import Config_IKP
from common.display import display_ikp


def paik():
    config = Config_IKP()
    solver_param = PANDA_PAIK
    solver_param.workdir = config.workdir
    solver_param.select_reference_posture_method = (
        config.method_of_select_reference_posture
    )
    solver = Solver(solver_param=solver_param)

    (l2, ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
    )  # type: ignore
    display_ikp(l2, ang, avg_inference_time)  # type: ignore
    print(
        f"success rate {config.method_of_select_reference_posture}: {np.round(success_rate, decimals=2)}")

def nsf():
    config = Config_IKP()
    solver_param = PANDA_NSF
    solver_param.workdir = config.workdir
    solver = Solver(solver_param=solver_param)

    (l2, ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
    )  # type: ignore
    display_ikp(l2, ang, avg_inference_time)  # type: ignore
    print(f"success rate nsf: {np.round(success_rate, decimals=2)}")

if __name__ == "__main__":
    paik()
    # nsf()
