# Import required packages
from __future__ import annotations
import numpy as np
import torch
from klampt.model import trajectory
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from .file import load_numpy, save_numpy, save_pickle
from .settings import SolverConfig
from .solver import Solver


def max_joint_angle_change(qs: torch.Tensor | np.ndarray):
    if isinstance(qs, torch.Tensor):
        qs = qs.detach().cpu().numpy()
    return np.rad2deg(np.max(np.abs(np.diff(qs, axis=0))))


class PathFollower(Solver):
    def __init__(self, solver_param: SolverConfig) -> None:
        super().__init__(solver_param)

        self.J_knn = NearestNeighbors(n_neighbors=1).fit(self.J)
        save_pickle(f"{solver_param.weight_dir}/J_knn.pth", self.J_knn)

    def solve_path(
        self,
        J: np.ndarray,
        P: np.ndarray,
        num_sols: int = 500,
        std: float = 0.25,
        return_evaluation: bool = False,
    ):
        self.base_std = std
        J_hat = self.solve_batch(
            P,
            self.F[
                self.J_knn.kneighbors(
                    J, return_distance=False
                ).flatten()  # type: ignore
            ],
            num_sols=num_sols,
        )  # type: ignore
        if not return_evaluation:
            return J_hat
        l2_errs, ang_errs = self.evaluate_pose_error_J3d_P2d(
            J_hat, P, return_posewise_evalution=True
        )
        mjac_arr = np.array([max_joint_angle_change(qs) for qs in J_hat])
        ddjc = np.linalg.norm(J_hat - J, axis=-1).mean(axis=-1)
        return J_hat, l2_errs, ang_errs, mjac_arr, ddjc

    def sample_multiple_Jtraj_and_Ppath(
        self,
        load_time: str = datetime.now().strftime("%m%d%H%M%S"),
        num_steps=20,
        num_traj=100,
        seed=47,
    ):
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

        traj_dir = self.param.traj_dir + load_time + "/"

        Ppath_file_path = traj_dir + "Ppath.npy"
        Jtraj_file_path = traj_dir + "Jtraj.npy"

        P = load_numpy(file_path=Ppath_file_path)
        J = load_numpy(file_path=Jtraj_file_path)

        if P is None or J is None:
            J = np.empty((num_traj, num_steps, self.n))
            P = np.empty((num_traj, num_steps, self.m))

            for i in range(num_traj):
                endPoints, _ = self._robot.sample_joint_angles_and_poses(
                    n=2, return_torch=False
                )
                Jtraj = trajectory.Trajectory(milestones=endPoints)  # type: ignore
                J[i] = np.array([Jtraj.eval(i / num_steps) for i in range(num_steps)])
                P[i] = self._robot.forward_kinematics(J[i, :, 0 : self._robot.n_dofs])

            save_numpy(file_path=Jtraj_file_path, arr=J)
            save_numpy(file_path=Ppath_file_path, arr=P)
        return J, P
