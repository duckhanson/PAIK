# Import required packages
import numpy as np
from tabulate import tabulate
from tqdm import trange
from pafik.solver import Solver
from pafik.settings import DEFAULT_NSF, DEFULT_SOLVER

NUM_POSES = 3000  # 100
NUM_SOLS = 400  # 1000
BATCH_SIZE = 5000
SUCCESS_THRESHOLD = (5e-3, 2)
STD = 0.1
WORKDIR = "/home/luca/example_package/pafik"
USE_NSF_ONLY = False
METHOD_OF_SELECT_REFERENCE_POSTURE = "knn"


def ikp():
    solver_param = DEFAULT_NSF if USE_NSF_ONLY else DEFULT_SOLVER
    solver_param.workdir = WORKDIR
    solver_param.select_reference_posture_method = METHOD_OF_SELECT_REFERENCE_POSTURE
    solver = Solver(solver_param=solver_param)

    (avg_l2, avg_ang, avg_inference_time, success_rate) = solver.evaluate_ikp_iterative(
        NUM_POSES,
        NUM_SOLS,
        std=STD,
        batch_size=BATCH_SIZE,
        success_threshold=SUCCESS_THRESHOLD,
    )  # type: ignore
    print(
        tabulate(
            [[avg_l2, np.rad2deg(avg_ang), avg_inference_time, success_rate]],
            headers=[
                "avg_l2",
                "avg_ang",
                "avg_inference_time",
                f"success_rate ({METHOD_OF_SELECT_REFERENCE_POSTURE})",
            ],
        )
    )


if __name__ == "__main__":
    ikp()
