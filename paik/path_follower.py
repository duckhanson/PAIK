# Import required packages
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from tabulate import tabulate
from sklearn.neighbors import NearestNeighbors
from paik.utils import load_numpy, save_numpy
from paik.settings import SolverConfig, DEFAULT_SOLVER_PARAM_M7_NORM

from paik.solver import (
    Solver,
    max_joint_angle_change,
)


class PathFollower(Solver):
    def __init__(self, solver_param: SolverConfig) -> None:
        super().__init__(solver_param)

        self.JP_knn = NearestNeighbors(n_neighbors=1).fit(
            np.column_stack([self._J_tr, self._P_tr])
        )

    def solve_path(
        self,
        J: np.ndarray,
        P: np.ndarray,
        num_traj: int = 500,
        return_numpy: bool = False,
        return_evaluation: bool = False,
    ):
        J_hat = self.solve(P, self._F[self.JP_knn.kneighbors(np.column_stack([J, P]), return_distance=False).flatten()], num_sols=num_traj, return_numpy=return_numpy)  # type: ignore
        if not return_evaluation:
            return J_hat

        l2_errs, ang_errs = self.evaluate_solutions(J_hat, P)
        mjac_arr = np.array([max_joint_angle_change(qs) for qs in J_hat])
        ddjc = np.linalg.norm(J_hat - J, axis=-1).mean(axis=-1)
        return J_hat, l2_errs, ang_errs, mjac_arr, ddjc

    def sample_Jtraj_Ppath(self, load_time: str = "", num_steps=20, seed=47):
        """
        sample a path from P_ts

        Parameters
        ----------
        load_time : str, optional
            file name of load P, by default ""
        num_steps : int, optional
            length of the generated path, by default 20

        Returns
        -------
        np.ndarray
            array_like(num_steps, m)
        """
        np.random.seed(seed)

        if load_time == "":
            traj_dir = self.param.traj_dir + datetime.now().strftime("%m%d%H%M%S") + "/"
        else:
            traj_dir = self.param.traj_dir + load_time + "/"

        Ppath_file_path = traj_dir + "Ppath.npy"
        Jtraj_file_path = traj_dir + "Jtraj.npy"

        P = load_numpy(file_path=Ppath_file_path)
        J = load_numpy(file_path=Jtraj_file_path)

        if len(P) == 0 or len(J) == 0:
            # endPoints = np.random.rand(2, cfg.m) # 2 for begin and end
            rand_idxs = np.random.randint(low=0, high=len(self._J_tr), size=2)
            endPoints = self._J_tr[rand_idxs]
            Jtraj = trajectory.Trajectory(milestones=endPoints)  # type: ignore
            J = np.array([Jtraj.eval(i / num_steps) for i in range(num_steps)])
            P = self._robot.forward_kinematics(J[:, 0 : self._robot.n_dofs])

            save_numpy(file_path=Jtraj_file_path, arr=J)
            save_numpy(file_path=Ppath_file_path, arr=P)
        return J, P
