# Import required packages
from __future__ import annotations
from typing import Any, Tuple
from os.path import isdir
from time import time
import numpy as np
import pandas as pd
import torch

from hnne import HNNE
from sklearn.neighbors import NearestNeighbors

from tqdm import trange

from .settings import SolverConfig, DEFULT_SOLVER
from .model import get_flow_model, get_robot
from .file import load_numpy, save_numpy, save_pickle, load_pickle
from .evaluate import evaluate_pose_error_P2d_P2d
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional


class Solver:
    def __init__(self, solver_param: SolverConfig = DEFULT_SOLVER) -> None:
        self.__solver_param = solver_param
        self._robot = get_robot(
            solver_param.robot_name, robot_dirs=solver_param.dir_paths
        )
        self._method_of_select_reference_posture = (
            solver_param.select_reference_posture_method
        )
        self._device = solver_param.device
        self._use_nsf_only = solver_param.use_nsf_only
        # Neural spline flow (NSF) with 3 sample features and 5 context features
        self.n, self.m, self.r = solver_param.n, solver_param.m, solver_param.r
        self.J, self.P, self.F = self.__load_training_data()

        self._solver, self._optimizer, self._scheduler = get_flow_model(
            solver_param
        )  # type: ignore
        self._base_std = solver_param.base_std
        # load inference data
        assert (
            self.n == self._robot.n_dofs
        ), f"n should be {self._robot.n_dofs} as the robot"

        path_p_knn = f"{solver_param.weight_dir}/P_knn.pth"
        try:
            self.P_knn = load_pickle(path_p_knn)
        except:
            self.P_knn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(self.P)
            save_pickle(
                path_p_knn,
                self.P_knn,
            )

    @property
    def base_std(self):
        return self._base_std

    @property
    def robot(self):
        return self._robot

    @property
    def param(self):
        return self.__solver_param

    @base_std.setter
    def base_std(self, value: float):
        assert value >= 0, "base_std should be greater than or equal to 0."
        self._base_std = value
        self.__change_solver_base()

    # private methods
    def __load_training_data(self):
        """
        Load training data from the given path, if not found, generate and save it.

        Returns:
            Tuple: J, P, F
        """
        assert isdir(
            self.param.train_dir
        ), f"{self.param.train_dir} not found, please change workdir to the project root!"
        data_path = (
            lambda name: f"{self.param.train_dir}/{name}-{self.param.N}-{self.n}-{self.m}-{self.r}.npy"
        )

        input_name_list = ["J", "P", "F"]
        J, P, F = [
            load_numpy(file_path=data_path(name)) for name in input_name_list
        ]

        if J is None or P is None:
            J, P = self._robot.sample_joint_angles_and_poses(
                n=self.param.N, return_torch=False
            )
            save_numpy(file_path=data_path("J"), arr=J)
            save_numpy(file_path=data_path("P"), arr=P)

        if F is None:
            assert self.r > 0, "r should be greater than 0."
            # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
            hnne = HNNE(dim=self.r)
            # maximum number of data for hnne (11M), we use max_num_data_hnne to test
            num_data = min(self.param.max_num_data_hnne, len(J))
            F = hnne.fit_transform(X=J[:num_data], dim=self.r, verbose=True)
            # query nearest neighbors for the rest of J
            if len(F) != len(J):
                knn = NearestNeighbors(n_neighbors=1).fit(J[:num_data])
                F = np.row_stack(
                    (
                        F,
                        F[
                            knn.kneighbors(
                                J[num_data:], n_neighbors=1, return_distance=False
                            ).flatten()  # type: ignore
                        ],
                    )
                )  # type: ignore

            save_numpy(file_path=data_path("F"), arr=F)
        print(f"[SUCCESS] F load from {data_path('F')}")

        # for normalization
        C = np.column_stack((P, F))

        self.__normalization_elements = {
            "J": {"mean": J.mean(axis=0), "std": J.std(axis=0)},
            "C": {
                # extra column for tuning std of noise for training data
                "mean": np.concatenate((C.mean(axis=0), np.zeros((1)))),
                "std": np.concatenate((C.std(axis=0), np.ones((1)))),
            },
        }

        return J, P, F

    def __change_solver_base(self):
        """
        Change the base distribution of the solver to the new base distribution.
        """
        self._solver = Flow(
            transforms=self._solver.transforms,  # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,), device=self._device),
                torch.ones((self._robot.n_dofs,), device=self._device) * self._base_std,
                buffer=True,
            ),  # type: ignore
        )

    # public methods
    def normalize_input_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Normalize input data. J_norm = (J - J_mean) / J_std, C_norm = (C - C_mean) / C_std

        Args:
            data (np.ndarray): input only support J and C
            name (str): name of the input data, only support "J" and "C"

        Returns:
            np.ndarray: normalized data
        """
        return (
            data - self.__normalization_elements[name]["mean"]
        ) / self.__normalization_elements[name]["std"]

    def denormalize_output_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """
        Denormalize output data. J = J_norm * J_std + J_mean, C = C_norm * C_std + C_mean

        Args:
            data (np.ndarray): input only support J_norm and C_norm
            name (str): name of the input data, only support "J" and "C"

        Returns:
            np.ndarray: denormalized data
        """
        return (
            data * self.__normalization_elements[name]["std"]
            + self.__normalization_elements[name]["mean"]
        )

    def remove_posture_feature(self, C: np.ndarray):
        """
        Remove posture feature from C

        Args:
            C (np.ndarray): conditions

        Returns:
            np.ndarray: conditions without posture feature
        """
        assert self._use_nsf_only and isinstance(C, np.ndarray)
        print("before remove posture feature", C.shape)
        # delete the last 2 of the last dimension of C, which is posture feature
        C = np.delete(C, -2, -1)
        print("after remove posture feature", C.shape)
        return C

    def solve(self, P: np.ndarray, F: np.ndarray, num_sols: int) -> np.ndarray:
        """
        Solve inverse kinematics problem.

        Args:
            P (np.ndarray): poses as least 2D array
            F (np.ndarray): posture features as least 2D array
            num_sols (int): number of solutions

        Returns:
            np.ndarray: J with shape (num_sols, num_poses, num_dofs)
        """
        C = self.normalize_input_data(
            np.column_stack((P, F, np.zeros((len(F), 1)))), "C"
        )
        C = self.remove_posture_feature(C) if self._use_nsf_only else C
        C = torch.from_numpy(C.astype(np.float32)).to(self._device)
        with torch.inference_mode():
            J = self._solver(C).sample((num_sols,))
        return self.denormalize_output_data(J.detach().cpu().numpy(), "J")

    def make_divisible_C(
        self, C: np.ndarray, batch_size: int
    ) -> tuple[np.ndarray, int]:
        """
        Make the number of conditions divisible by batch_size. Reapeat the last few conditions to make it divisible.

        Args:
            C (np.ndarray): conditions
            batch_size (int): batch size

        Returns:
            Tuple[np.ndarray, int]: divisible conditions and the number of complementary conditions
        """
        assert C.ndim == 2
        complementary = batch_size - len(C) % batch_size
        complementary = 0 if complementary == batch_size else complementary
        C = np.concatenate((C, C[:complementary]), axis=0) if complementary > 0 else C
        return C, complementary

    def remove_complementary_J(self, J: np.ndarray, complementary: int) -> np.ndarray:
        """
        Remove the complementary conditions from J.

        Args:
            J (np.ndarray): complemented J
            complementary (int): number of complementary conditions

        Returns:
            np.ndarray: J without complementary elements
        """
        assert J.ndim == 2
        return J[:-complementary] if complementary > 0 else J

    def solve_batch(
        self,
        P: np.ndarray,
        F: np.ndarray,
        num_sols: int,
        batch_size: int = 4000,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Solve inverse kinematics problem in batch.

        Args:
            P (np.ndarray): poses with shape (num_poses, m)
            F (np.ndarray): F with shape (num_poses, r)
            num_sols (int): number of solutions
            batch_size (int, optional): batch size. Defaults to 4000.
            verbose (bool, optional): use trange or not. Defaults to True.

        Returns:
            np.ndarray: J with shape (num_sols, num_poses, num_dofs)
        """

        if len(P) * num_sols < batch_size:
            return self.solve(P, F, num_sols)

        # shape: (num_poses, C.shape[-1] = m + r + 1)
        C = np.column_stack((P, F, np.zeros((len(F), 1))))
        C = self.normalize_input_data(C, "C")
        # C: (num_poses, m + r + 1) -> C: (num_sols * num_poses, m + r + 1)
        C = np.tile(C, (num_sols, 1))
        C = self.remove_posture_feature(C) if self._use_nsf_only else C
        C, complementary = self.make_divisible_C(C, batch_size)
        # shape: ((num_sols * num_poses + complementary) // batch_size, batch_size, C.shape[-1])
        C = torch.from_numpy(
            C.astype(np.float32).reshape(-1, batch_size, C.shape[-1])
        ).to(self._device)

        J = torch.empty((len(C), batch_size, self._robot.n_dofs), device=self._device)
        iterator = trange(len(C)) if verbose else range(len(C))
        with torch.inference_mode():
            for i in iterator:
                J[i] = self._solver(C[i]).sample()

        J = J.detach().cpu().numpy()
        J = self.remove_complementary_J(
            J.reshape(-1, self._robot.n_dofs), complementary
        )
        return self.denormalize_output_data(
            J.reshape(num_sols, -1, self._robot.n_dofs), "J"
        )

    def evaluate_pose_error_J3d_P2d(
        self,
        J: np.ndarray,
        P: np.ndarray,
        return_posewise_evalution: bool = False,
        return_all: bool = False,
    ) -> tuple[Any, Any]:
        """
        Evaluate pose error given generated joint configurations J and ground truth poses P. Return default is l2 and ang with shape (1).

        Args:
            J (np.ndarray): generated joint configurations with shape (num_sols, num_poses, num_dofs)
            P (np.ndarray): ground truth poses with shape (num_poses, m)
            return_posewise_evalution (bool, optional): return l2 and ang with shape (num_poses,). Defaults to False.
            return_all (bool, optional): return l2 and ang with shape (num_sols * num_poses). Defaults to False.

        Returns:
            tuple[Any, Any]: l2 and ang, default shape (1), posewise evaluation with shape (num_poses,), or all evaluation with shape (num_sols * num_poses)
        """
        num_poses, num_sols = len(P), len(J)
        assert len(J.shape) == 3 and len(P.shape) == 2 and J.shape[1] == num_poses

        # P: (num_poses, m), P_expand: (num_sols * num_poses, m)
        P_expand = np.tile(P, (num_sols, 1))

        P_hat = self.robot.forward_kinematics(J.reshape(-1, self.n))
        l2, ang = evaluate_pose_error_P2d_P2d(P_hat, P_expand)  # type: ignore

        if return_posewise_evalution:
            return (
                l2.reshape(num_sols, num_poses).mean(axis=1),
                ang.reshape(num_sols, num_poses).mean(axis=1),
            )
        elif return_all:
            return l2, ang
        return l2.mean(), ang.mean()

    def select_reference_posture(self, P: np.ndarray):
        if self._method_of_select_reference_posture == "knn":
            # type: ignore
            return self.F[
                self.P_knn.kneighbors(
                    np.atleast_2d(P), n_neighbors=1, return_distance=False
                ).flatten()  # type: ignore
            ]
        elif self._method_of_select_reference_posture == "random":
            min_F, max_F = np.min(self.F), np.max(self.F)
            return np.random.rand(len(P), self.r) * (max_F - min_F) + min_F
        elif self._method_of_select_reference_posture == "pick":
            # randomly pick one posture from train set
            return self.F[np.random.randint(0, len(self.F), len(P))]
        else:
            raise NotImplementedError

    def evaluate_ikp_iterative(
        self,
        num_poses: int,
        num_sols: int,
        batch_size: int = 5000,
        std: float = 0.25,
        success_threshold: Tuple[float, float] = (1e-4, 1e-4),
        verbose: bool = True,
    ):  # -> tuple[Any, Any, float] | tuple[Any, Any]:# -> tuple[Any, Any, float] | tuple[Any, Any]:
        self.base_std = std
        # Randomly sample poses from test set
        _, P = self._robot.sample_joint_angles_and_poses(
            n=num_poses, return_torch=False
        )
        time_begin = time()
        # Data Preprocessing
        F = self.select_reference_posture(P)

        # Begin inference
        J_hat = self.solve_batch(P, F, num_sols, batch_size=batch_size, verbose=verbose)

        l2, ang = self.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        df = pd.DataFrame({"l2": l2, "ang": np.rad2deg(ang)})
        print(df.describe())

        return tuple(
            [
                l2.mean(),
                ang.mean(),
                avg_inference_time,
                round(
                    len(
                        df.query(
                            f"l2 < {success_threshold[0]} & ang < {success_threshold[1]}"
                        )
                    )
                    / (num_poses * num_sols),
                    3,
                ),
            ]
        )
