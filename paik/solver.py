# Import required packages
from __future__ import annotations
from typing import Any, Tuple
from time import time
import numpy as np
import pandas as pd
import torch

from hnne import HNNE
from sklearn.neighbors import NearestNeighbors

from tqdm import trange

from paik.settings import SolverConfig, DEFULT_SOLVER
from paik.model import get_flow_model, get_robot
from paik.file import load_numpy, save_numpy, save_pickle, load_pickle
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional


class Solver:
    def __init__(
        self,
        solver_param: SolverConfig = DEFULT_SOLVER,
    ) -> None:
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
        self._n, self._m, self._r = solver_param.n, solver_param.m, solver_param.r
        self._solver, self._optimizer, self._scheduler = get_flow_model(
            solver_param)  # type: ignore
        self._base_std = solver_param.base_std
        self._init_latent = torch.zeros((self.robot.n_dofs)).to(self._device)
        # load inference data
        assert (
            self._n == self._robot.n_dofs
        ), f"n should be {self._robot.n_dofs} as the robot"

        self._J_tr, self._P_tr, self._F = self.__load_training_data()

        try:
            self.nearest_neighnbor_P = load_pickle(
                "./weights/panda/nearest_neighnbor_P.pth"
            )
        except:
            self.nearest_neighnbor_P = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
                self._P_tr
            )
            save_pickle(
                "./weights/panda/nearest_neighnbor_P.pth", self.nearest_neighnbor_P
            )

    @property
    def latent(self):
        return self._init_latent

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
        assert value >= 0 and value < 1
        self._base_std = value
        self.__update_solver()

    @latent.setter
    def latent(self, value: torch.Tensor):
        assert value.shape == (self._robot.n_dofs)
        self._init_latent = value
        self.__update_solver()

    # private methods
    def __load_training_data(self):
        path_J = (
            f"{self.param.train_dir}/J-{self.param.N}-{self._n}-{self._m}-{self._r}.npy"
        )
        path_P = (
            f"{self.param.train_dir}/P-{self.param.N}-{self._n}-{self._m}-{self._r}.npy"
        )
        J = load_numpy(file_path=path_J)
        P = load_numpy(file_path=path_P)

        if len(J) != self.param.N or len(P) != self.param.N:
            J, P = self._robot.sample_joint_angles_and_poses(
                n=self.param.N, return_torch=False
            )
            save_numpy(file_path=path_J, arr=J)
            save_numpy(file_path=path_P, arr=P[:, : self._m])

        assert self._r > 0
        path_F = f"{self.param.train_dir}/F-{self.param.N}-{self._n}-{self._m}-{self._r}-from-C-space.npy"
        F = load_numpy(file_path=path_F)

        if F.shape != (len(J), self._r):
            # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
            hnne = HNNE(dim=self._r)
            # maximum number of data for hnne (11M), we use max_num_data_hnne to test
            num_data = min(self.param.max_num_data_hnne, len(J))
            F = hnne.fit_transform(X=J[:num_data], dim=self._r, verbose=True)
            # query nearest neighbors for the rest of J
            if len(F) != len(J):
                knn = NearestNeighbors(n_neighbors=1)
                knn.fit(J[:num_data])
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

            save_numpy(file_path=path_F, arr=F)
        print(f"[SUCCESS] F load from {path_F}")

        # for normalization
        self.__mean_J, self.__std_J = J.mean(axis=0), J.std(axis=0)
        C = np.column_stack((P, F))
        self.__mean_C = np.concatenate((C.mean(axis=0), np.zeros((1))))
        std_C = np.concatenate((C.std(axis=0), np.ones((1))))
        scale = np.ones_like(self.__mean_C)
        scale[self._m: self._m + self._r] *= self.__solver_param.posture_feature_scale
        self.__std_C = std_C / scale

        return J, P, F

    def __update_solver(self):
        self._solver = Flow(
            transforms=self._solver.transforms,  # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,), device=self._device)
                + self._init_latent,
                torch.ones((self._robot.n_dofs,), device=self._device)
                * self._base_std,
                buffer=True,
            ),  # type: ignore
        )

    # public methods
    def norm_J(self, J: np.ndarray):
        assert isinstance(J, np.ndarray)
        return (J - self.__mean_J) / self.__std_J

    def norm_C(self, C: np.ndarray):
        assert isinstance(C, np.ndarray)
        return (C - self.__mean_C) / self.__std_C

    def denorm_J(self, J: np.ndarray):
        assert isinstance(J, np.ndarray)
        return J * self.__std_J + self.__mean_J

    def denorm_C(self, C: np.ndarray):
        assert isinstance(C, np.ndarray)
        return C * self.__std_C + self.__mean_C

    def remove_posture_feature(self, C: np.ndarray):
        assert self._use_nsf_only and isinstance(C, np.ndarray)
        print("before remove posture feature", C.shape)
        if len(C.shape) == 2:
            C = np.column_stack((C[:, : self._m], C[:, -1]))
        elif len(C.shape) == 3:
            C = np.concatenate((C[:, :, : self._m], C[:, :, -1:]), axis=-1)
        print("after remove posture feature", C.shape)
        return C

    def solve(self, P: np.ndarray, F: np.ndarray, num_sols: int):
        C = self.norm_C(np.column_stack((P, F, np.zeros((len(F), 1)))))
        C = self.remove_posture_feature(C) if self._use_nsf_only else C
        C = torch.from_numpy(C.astype(np.float32)).to(self._device)
        with torch.inference_mode():
            J = self._solver(C).sample((num_sols,))
        return self.denorm_J(J.detach().cpu().numpy())

    def solve_batch(
        self,
        P: np.ndarray,
        F: np.ndarray,
        num_sols: int,
        batch_size: int = 4000,
        verbose: bool = True,
    ):
        if len(P) * num_sols < batch_size:
            return self.solve(P, F, num_sols)
        C = self.norm_C(
            np.repeat(
                np.expand_dims(np.column_stack(
                    (P, F, np.zeros((len(F), 1)))), axis=0),
                num_sols,
                axis=0,
            )
        )
        C = self.remove_posture_feature(C) if self._use_nsf_only else C
        C = C.reshape(-1, C.shape[-1])
        complementary = batch_size - len(C) % batch_size
        complementary = 0 if complementary == batch_size else complementary
        C = np.concatenate((C, C[:complementary]),
                           axis=0) if complementary > 0 else C
        C = C.reshape(-1, batch_size, C.shape[-1])
        C = torch.from_numpy(C.astype(np.float32)).to(self._device)
        J = torch.empty((len(C), batch_size, self._robot.n_dofs),
                        device=self._device)

        if verbose:
            with torch.inference_mode():
                for i in trange(len(C)):
                    J[i] = self._solver(C[i]).sample()
        else:
            with torch.inference_mode():
                for i in range(len(C)):
                    J[i] = self._solver(C[i]).sample()

        J = J.detach().cpu().numpy()
        J = (
            J.reshape(-1, self._robot.n_dofs)[:-complementary]
            if complementary > 0
            else J
        )
        return self.denorm_J(J.reshape(num_sols, -1, self._robot.n_dofs))

    def pose_error_evalute(
        self,
        J: np.ndarray,
        P: np.ndarray,
        return_posewise_evalution: bool = False,
        return_all: bool = False,
    ) -> tuple[Any, Any]:
        def geometric_distance_between_quaternions(
            q1: np.ndarray, q2: np.ndarray
        ) -> np.ndarray:
            # from jrl.conversions
            # Note: Decreasing this value to 1e-8 greates NaN gradients for nearby quaternions.
            acos_clamp_epsilon = 1e-7
            dot = np.clip(np.sum(q1 * q2, axis=1), -1, 1)
            # Note: Updated by @jstmn on Feb24 2023
            distance = 2 * np.arccos(
                np.clip(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon)
            )
            # distance = 2 * np.arccos(dot)
            distance = np.abs(np.remainder(
                distance + np.pi, 2 * np.pi) - np.pi)
            return distance

        num_poses = len(P)
        num_sols = len(J)
        J = np.expand_dims(J, axis=1) if len(J.shape) == 2 else J
        assert J.shape == (num_sols, num_poses, self._robot.n_dofs)

        P_expand = np.repeat(np.expand_dims(P, axis=0), len(J), axis=0).reshape(
            -1, P.shape[-1]
        )
        P_hat = self._robot.forward_kinematics(J.reshape(-1, self._n)).reshape(
            -1, P.shape[-1]
        )
        l2 = np.linalg.norm(P_expand[:, :3] - P_hat[:, :3], axis=1)
        ang = geometric_distance_between_quaternions(
            P_expand[:, 3:], P_hat[:, 3:]
        )  # type: ignore

        if return_posewise_evalution:
            return l2.reshape(num_sols, num_poses).mean(axis=1), ang.reshape(
                num_sols, num_poses
            ).mean(axis=1)
        elif return_all:
            return l2, ang
        return l2.mean(), ang.mean()

    def select_reference_posture(self, P: np.ndarray):
        if self._method_of_select_reference_posture == "knn":
            # type: ignore
            return self._F[
                self.nearest_neighnbor_P.kneighbors(
                    np.atleast_2d(P), n_neighbors=1, return_distance=False
                ).flatten()
            ]
        elif self._method_of_select_reference_posture == "random":
            mF, MF = np.min(self._F), np.max(self._F)
            return np.random.rand(len(P), self._r) * (MF - mF) + mF
        elif self._method_of_select_reference_posture == "pick":
            # randomly pick one posture from train set
            return self._F[np.random.randint(0, len(self._F), len(P))]
        else:
            raise NotImplementedError

    def ikp_iterative_evalute(
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
        J_hat = self.solve_batch(
            P, F, num_sols, batch_size=batch_size, verbose=verbose)

        l2, ang = self.pose_error_evalute(J_hat, P, return_all=True)
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
