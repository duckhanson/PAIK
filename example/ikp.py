# Import required packages
import time
import numpy as np
from tqdm import trange
import torch
from paik.solver import Solver
from paik.settings import (
    PANDA_NSF,
    PANDA_PAIK,
    FETCH_PAIK,
    FETCH_ARM_PAIK,
    IIWA7_PAIK,
    ATLAS_ARM_PAIK,
    ATLAS_WAIST_ARM_PAIK,
    BAXTER_ARM_PAIK,
    PR2_PAIK
)

from common.config import Config_IKP
from common.display import display_ikp, save_ikp


def paik():
    config = Config_IKP()
    solver = Solver(solver_param=ATLAS_WAIST_ARM_PAIK,
                    load_date='best', work_dir=config.workdir)

    (l2, ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
        select_reference=config.select_reference,
    )  # type: ignore
    
    save_ikp(config.record_dir, "paik", l2, ang, avg_inference_time) # type: ignore


def nsf():
    config = Config_IKP()
    solver = Solver(solver_param=PANDA_NSF,
                    load_date='best', work_dir=config.workdir)

    (l2, ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        config.num_poses,
        config.num_sols,
        std=config.std,
        batch_size=config.batch_size,
        success_threshold=config.success_threshold,
    )  # type: ignore

    save_ikp(config.record_dir, "nsf", l2, ang, avg_inference_time) # type: ignore

if __name__ == "__main__":
    paik()
    # nsf()
