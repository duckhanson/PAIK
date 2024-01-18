import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from paik.solver import Solver
from paik.train import init_seeds
from paik.settings import (
    DEFAULT_SOLVER_PARAM_M7_NORM,
    DEFAULT_SOLVER_PARAM_M7_DISABLE_POSTURE_FEATURES,
)

NUM_POSES = 1
NUM_SOLUTIONS = 10000


def posture_sweep():
    solver = Solver(solver_param=DEFAULT_SOLVER_PARAM_M7_NORM)
    J, P, F = solver.get_random_JPF(num_samples=NUM_POSES)

    solver.shrink_ratio = 0.0

    avg_pos, avg_ori = solver.random_sample_solutions_with_evaluation(  # type: ignore
        num_poses=100, num_sols=100
    )

    print(tabulate([["avg_pos", avg_pos], ["avg_ori", avg_ori]]))

    l2_errs = np.zeros((NUM_POSES, NUM_SOLUTIONS))
    ang_errs = np.zeros((NUM_POSES, NUM_SOLUTIONS))
    config_errs = np.zeros((NUM_POSES, NUM_SOLUTIONS, 7))
    config = np.empty((NUM_POSES, NUM_SOLUTIONS, 7))

    pivot = np.random.randint(0, NUM_SOLUTIONS)

    for i, p in enumerate(P):
        F = np.random.rand(NUM_SOLUTIONS, 1)
        p = np.tile(p, (NUM_SOLUTIONS, 1))
        J = solver.solve(P=p, F=F, num_sols=1, return_numpy=True)
        config[i] = J.reshape((NUM_SOLUTIONS, 7))
        l2_errs[i], ang_errs[i] = solver.evaluate_solutions(J, p, return_col=True)
        config_errs[i] = np.abs(config[i] - config[i, pivot])

    df = pd.DataFrame(
        {
            "posture": F.flatten(),
            "l2_errs": l2_errs.flatten(),
            "ang_errs": ang_errs.flatten(),
            "config_errs": config_errs.mean(-1).flatten(),
        }
    )

    print(df.describe())
    query_df = df.query(f"l2_errs < {avg_pos}")
    print(query_df.describe())

    ax = query_df["config_errs"].plot.hist(bins=30)
    ax.set_xlabel("Config Error (rads)")
    plt.show()

    ax2 = query_df.plot.scatter(x="config_errs", y="posture")
    ax2.set_xlabel("Config Error (rads)")
    ax2.set_ylabel("posture features")
    plt.show()


if __name__ == "__main__":
    posture_sweep()
