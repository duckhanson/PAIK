# Import required packages
from __future__ import annotations
from typing import Tuple
import os
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from hnne import HNNE
from sklearn.neighbors import NearestNeighbors

from jrl.evaluation import _get_target_pose_batch
from jrl.conversions import geodesic_distance_between_quaternions

from paik.settings import SolverConfig, DEFAULT_SOLVER_PARAM_M7_NORM
from paik.model import get_flow_model, get_robot
from paik.file import load_numpy, save_numpy, save_pickle
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional


class Solver:
    def __init__(
        self,
        solver_param: SolverConfig = DEFAULT_SOLVER_PARAM_M7_NORM,
    ) -> None:
        self.__solver_param = solver_param
        self._robot = get_robot(
            solver_param.robot_name, robot_dirs=solver_param.dir_paths
        )
        self._extract_posture_feature_from_C_space = (
            solver_param.extract_posture_feature_from_C_space
        )
        self._method_of_select_reference_posture = (
            solver_param.method_of_select_reference_posture
        )
        self._device = solver_param.device
        self._disable_posture_feature = solver_param.disable_posture_feature
        # Neural spline flow (NSF) with 3 sample features and 5 context features
        self._n, self._m, self._r = solver_param.nmr
        self._solver, self._optimizer, self._scheduler = get_flow_model(
            enable_load_model=solver_param.enable_load_model,
            num_transforms=solver_param.num_transforms,
            subnet_width=solver_param.subnet_width,
            subnet_num_layers=solver_param.subnet_num_layers,
            shrink_ratio=solver_param.shrink_ratio,
            lr=solver_param.lr,
            lr_weight_decay=solver_param.lr_weight_decay,
            gamma=solver_param.gamma,
            device=self._device,
            model_architecture=solver_param.model_architecture,
            random_perm=solver_param.random_perm,
            path_solver=f"{solver_param.weight_dir}/{solver_param.ckpt_name}.pth",
            n=self._n,
            m=self._m,
            r=self._r,
            shce_patience=solver_param.shce_patience,
            disable_posture_feature=self._disable_posture_feature,
        )  # type: ignore
        self._shrink_ratio = solver_param.shrink_ratio
        self._init_latent = torch.zeros((1, self.robot.n_dofs)).to(self._device)
        # load inference data
        assert (
            self._n == self._robot.n_dofs
        ), f"n should be {self._robot.n_dofs} as the robot"

        self._J_tr, self._P_tr, self._P_ts, self._F = self.__load_all_data()
        self.nearest_neighnbor_P = NearestNeighbors(n_neighbors=1).fit(self._P_tr)
        # save_pickle("./weights/panda/nearest_neighnbor_P.pth", self.nearest_neighnbor_P)

        self._enable_normalize = solver_param.enable_normalize
        if self._enable_normalize:
            self.__mean_J, self.__std_J = self._J_tr.mean(axis=0), self._J_tr.std(
                axis=0
            )
            C = np.column_stack((self._P_tr, self._F))
            self.__mean_C = np.concatenate((C.mean(axis=0), np.zeros((1))))
            std_C = np.concatenate((C.std(axis=0), np.ones((1))))
            scale = np.ones_like(self.__mean_C)
            scale[self._m : self._m + self._r] *= solver_param.posture_feature_scale
            self.__std_C = std_C / scale

    @property
    def latent(self):
        return self._init_latent

    @property
    def shrink_ratio(self):
        return self._shrink_ratio

    @property
    def robot(self):
        return self._robot

    @property
    def param(self):
        return self.__solver_param

    @shrink_ratio.setter
    def shrink_ratio(self, value: float):
        assert value >= 0 and value < 1
        self._shrink_ratio = value
        self.__update_solver()

    @latent.setter
    def latent(self, value: torch.Tensor):
        assert value.shape == (1, self._robot.n_dofs)
        self._init_latent = value
        self.__update_solver()

    # private methods
    def __load_all_data(self):
        def get_JP_data(train: bool):
            N = self.param.N_train if train else self.param.N_test
            _dir = self.param.train_dir if train else self.param.val_dir
            path_J = f"{_dir}/J-{N}-{self._n}-{self._m}-{self._r}.npy"
            path_P = f"{_dir}/P-{N}-{self._n}-{self._m}-{self._r}.npy"

            J = load_numpy(file_path=path_J)
            P = load_numpy(file_path=path_P)

            if len(J) != N or len(P) != N:
                J, P = self._robot.sample_joint_angles_and_poses(
                    n=N, return_torch=False
                )
                save_numpy(file_path=path_J, arr=J)
                save_numpy(file_path=path_P, arr=P[:, : self._m])

            return J, P

        def get_posture_feature(J: np.ndarray, P: np.ndarray):
            assert self._r > 0
            file_path = (
                f"{self.param.train_dir}/F-{self.param.N_train}-{self._n}-{self._m}-{self._r}-from-C-space.npy"
                if self._extract_posture_feature_from_C_space
                else f"{self.param.train_dir}/F-{self.param.N_train}-{self._n}-{self._m}-{self._r}.npy"
            )
            F = load_numpy(file_path=file_path)

            GENERATE_NEW = F.shape != (len(J), self._r)
            if GENERATE_NEW:
                # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
                hnne = HNNE(dim=self._r)
                # maximum number of data for hnne (11M), we use max_num_data_hnne to test
                num_data = min(self.param.max_num_data_hnne, len(J))
                S = (
                    J
                    if self._extract_posture_feature_from_C_space
                    else np.column_stack((J, P))
                )
                F = hnne.fit_transform(X=S[:num_data], dim=self._r, verbose=True)
                # query nearest neighbors for the rest of J
                if len(F) != len(J):
                    knn = NearestNeighbors(n_neighbors=1)
                    knn.fit(S[:num_data])
                    F = np.row_stack(
                        (
                            F,
                            F[
                                knn.kneighbors(
                                    S[num_data:], n_neighbors=1, return_distance=False
                                ).flatten()  # type: ignore
                            ],
                        )
                    )  # type: ignore

                save_numpy(file_path=file_path, arr=F)
            print(f"F load successfully from {file_path}")

            return F

        J_train, P_train = get_JP_data(train=True)
        _, P_test = get_JP_data(train=False)
        F = get_posture_feature(J=J_train, P=P_train)
        return J_train, P_train, P_test, F

    def __update_solver(self):
        self._solver = Flow(
            transforms=self._solver.transforms,  # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,), device=self._device)
                + self._init_latent,
                torch.ones((self._robot.n_dofs,), device=self._device)
                * self._shrink_ratio,
                buffer=True,
            ),  # type: ignore
        )

    # public methods
    def norm_J(self, J: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()
        return torch.from_numpy(((J - self.__mean_J) / self.__std_J).astype(np.float32))

    def norm_C(self, C: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        return torch.from_numpy(((C - self.__mean_C) / self.__std_C).astype(np.float32))  # type: ignore

    def denorm_J(self, J: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()
        return torch.from_numpy((J * self.__std_J + self.__mean_J).astype(np.float32))

    def denorm_C(self, C: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        return torch.from_numpy((C * self.__std_C + self.__mean_C).astype(np.float32))  # type: ignore

    def remove_posture_feature(self, C: np.ndarray | torch.Tensor):
        assert self._disable_posture_feature
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        C = np.column_stack((C[:, : self._m], C[:, -1]))
        return torch.from_numpy(C.astype(np.float32))

    def solve(
        self, P: np.ndarray, F: np.ndarray, num_sols: int, return_numpy: bool = False
    ):
        C = np.column_stack((P, F, np.zeros((len(F), 1)))).astype(np.float32)
        C = self.norm_C(C) if self._enable_normalize else torch.from_numpy(C)
        C = self.remove_posture_feature(C) if self._disable_posture_feature else C
        C = C.to(self._device)
        with torch.inference_mode():
            J = self._solver(C).sample((num_sols,)).detach().cpu()

        if self._enable_normalize:
            J = self.denorm_J(J)

        return J.numpy() if return_numpy else J

    def get_random_JPF(self, num_samples: int):
        # Randomly sample poses from train set
        J, P = self._robot.sample_joint_angles_and_poses(
            n=num_samples, return_torch=False
        )
        F = self._F[self.nearest_neighnbor_P.kneighbors(np.atleast_2d(P), n_neighbors=1, return_distance=False).flatten()]  # type: ignore
        return J, P, F

    def evaluate_solutions(
        self,
        J: np.ndarray | torch.Tensor,
        P: np.ndarray,
        return_row: bool = False,
        return_col: bool = False,
        return_all: bool = False,
    ) -> tuple[Any, Any]:
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        def get_pose_errors(
            J_hat: np.ndarray,
            P: torch.Tensor | np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            P = _get_target_pose_batch(P, J_hat.shape[0])
            P_hat = self._robot.forward_kinematics(J_hat[:, : self._n])

            # Positional Error
            l2_errors = np.linalg.norm(P_hat[:, :3] - P[:, :3], axis=1)
            ang_errors = geodesic_distance_between_quaternions(P[:, 3:], P_hat[:, 3:])
            return l2_errors, ang_errors  # type: ignore

        num_poses = len(P)
        num_sols = len(J)
        J = np.expand_dims(J, axis=1) if len(J.shape) == 2 else J
        assert J.shape == (num_sols, num_poses, self._robot.n_dofs)

        l2_errs = np.empty((num_poses, num_sols))
        ang_errs = np.empty((num_poses, num_sols))
        for i in range(num_poses):
            l2_errs[i], ang_errs[i] = get_pose_errors(
                J_hat=J[:, i, :], P=P[i]  # type: ignore
            )
        if return_row:
            return l2_errs.mean(axis=0), ang_errs.mean(axis=0)
        elif return_col:
            return l2_errs.mean(axis=1), ang_errs.mean(axis=1)
        elif return_all:
            return l2_errs.flatten(), ang_errs.flatten()
        return l2_errs.mean(), ang_errs.mean()

    def select_reference_posture(self, P: np.ndarray):
        if self._method_of_select_reference_posture == "knn":
            return self._F[self.nearest_neighnbor_P.kneighbors(np.atleast_2d(P), n_neighbors=1, return_distance=False).flatten()]  # type: ignore
        elif self._method_of_select_reference_posture == "random":
            mF, MF = np.min(self._F), np.max(self._F)
            return np.random.rand(len(P), self._r) * (MF - mF) + mF
        elif self._method_of_select_reference_posture == "pick":
            # randomly pick one posture from train set
            return self._F[np.random.randint(0, len(self._F), len(P))]
        else:
            raise NotImplementedError

    def random_sample_solutions_with_evaluation(
        self,
        num_poses: int,
        num_sols: int,
        return_time: bool = False,
        return_success_rate: bool = False,
        success_threshold: Tuple[float, float] = (1e-4, 1e-4),
    ):  # -> tuple[Any, Any, float] | tuple[Any, Any]:# -> tuple[Any, Any, float] | tuple[Any, Any]:
        # Randomly sample poses from test set
        # P = self._P_ts[np.random.choice(self._P_ts.shape[0], num_poses, replace=False)]
        _, P = self._robot.sample_joint_angles_and_poses(
            n=num_poses, return_torch=False
        )
        time_begin = time()
        # Data Preprocessing
        F = self.select_reference_posture(P)

        # Begin inference
        J_hat = self.solve(P, F, num_sols, return_numpy=True)
        inference_time = round((time() - time_begin), 3)

        P = P if self._m == 7 else np.column_stack((P, np.ones(shape=(len(P), 4))))

        success_rate = 0
        if return_success_rate:
            l2_errs, ang_errs = self.evaluate_solutions(J_hat, P, return_all=True)
            avg_l2_errs, avg_ang_errs = l2_errs.mean(), ang_errs.mean()
            # success_rate = sum(np.where(l2_errs < success_threshold)) / (num_poses * num_sols)
            df = pd.DataFrame({"l2_errs": l2_errs, "ang_errs": np.rad2deg(ang_errs)})
            # use df to get success rate where l2_errs < 1e-4
            df_query = df.query(f"l2_errs < {success_threshold[0]} & ang_errs < {success_threshold[1]}")
            success_rate = df_query.count() / (
                num_poses * num_sols
            )
            print(df.describe())
        else:
            avg_l2_errs, avg_ang_errs = self.evaluate_solutions(J_hat, P)

        # errors_time = round((time() - time_begin), 3) - inference_time
        # print(
        #     tabulate(
        #         [[inference_time, errors_time]],
        #         headers=["inference time", "evaluation time"],
        #     )
        # )

        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        results = [avg_l2_errs, avg_ang_errs]
        if return_time:
            results.append(avg_inference_time)
        if return_success_rate:
            results.append(success_rate)

        return tuple(results)
