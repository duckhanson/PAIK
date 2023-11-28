# Import required packages
from __future__ import annotations
import os
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from hnne import HNNE
from sklearn.neighbors import NearestNeighbors

from klampt.model import trajectory
from jrl.robot import Robot
from jrl.evaluation import _get_target_pose_batch
from jrl.conversions import geodesic_distance_between_quaternions

from paik.settings import SolverConfig
from paik.model import get_flow_model, get_knn, get_robot
from paik.utils import load_numpy, save_numpy
from paik.dataset import nearest_neighbor_F
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional

DEFAULT_SOLVER_PARAM_M3 = SolverConfig(
    lr=0.00033,
    gamma=0.094,
    opt_type="adamw",
    noise_esp=0.001,
    batch_size=128,
    num_epochs=10,
    random_perm=True,
    subnet_width=1024,
    num_transforms=10,
    lr_weight_decay=0.013,
    noise_esp_decay=0.8,
    subnet_num_layers=3,
    model_architecture="nsf",
    shrink_ratio=0.61,
    ckpt_name="0930-0346",
    nmr=(7, 3, 4),
    enable_load_model=True,
    device="cuda",
)

DEFAULT_SOLVER_PARAM_M7 = SolverConfig(
    lr=0.00036,
    gamma=0.084,
    opt_type="adamw",
    noise_esp=0.0019,
    batch_size=128,
    num_epochs=15,
    random_perm=True,
    subnet_width=1150,
    num_transforms=8,
    lr_weight_decay=0.018,
    noise_esp_decay=0.92,
    subnet_num_layers=3,
    model_architecture="nsf",
    shrink_ratio=0.61,
    ckpt_name="1107-1013",
    nmr=(7, 7, 1),
    enable_load_model=True,
    device="cuda",
)

DEFAULT_SOLVER_PARAM_M7_NORM = SolverConfig(
    lr=0.00037,
    gamma=0.086,
    opt_type="adamw",
    noise_esp=0.0025,
    random_perm=False,
    shrink_ratio=0.68,
    subnet_width=1024,
    num_transforms=8,
    lr_weight_decay=0.012,
    noise_esp_decay=0.97,
    enable_normalize=True,
    subnet_num_layers=3,
    ckpt_name="1128-0459",  # "1128-0459", "1128-0857"
)


class Solver:
    def __init__(
        self,
        solver_param: SolverConfig = DEFAULT_SOLVER_PARAM_M7,
    ) -> None:
        self._solver_param = solver_param
        self._robot = get_robot(
            solver_param.robot_name, robot_dirs=solver_param.dir_paths
        )
        self._device = solver_param.device
        # Neural spline flow (NSF) with 3 sample features and 5 context features
        n, m, r = solver_param.nmr
        flow, optimizer, scheduler = get_flow_model(
            enable_load_model=solver_param.enable_load_model,
            num_transforms=solver_param.num_transforms,
            subnet_width=solver_param.subnet_width,
            subnet_num_layers=solver_param.subnet_num_layers,
            shrink_ratio=solver_param.shrink_ratio,
            lr=solver_param.lr,
            lr_weight_decay=solver_param.lr_weight_decay,
            gamma=solver_param.gamma,
            optimizer_type=solver_param.opt_type,
            device=self._device,
            model_architecture=solver_param.model_architecture,
            random_perm=solver_param.random_perm,
            path_solver=f"{solver_param.weight_dir}/{solver_param.ckpt_name}.pth",
            n=n,
            m=m,
            r=r,
        )  # type: ignore
        self._solver = flow
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._shrink_ratio = solver_param.shrink_ratio
        self._init_latent = torch.zeros((1, self.robot.n_dofs)).to(self._device)

        # load inference data
        assert n == self._robot.n_dofs, f"n should be {self._robot.n_dofs} as the robot"

        self._n, self._m, self._r = n, m, r
        self._J_tr, self._P_tr, self._P_ts, self._F = self.__load_all_data()
        self._knn = get_knn(
            P_tr=self._P_tr,
            path=f"{solver_param.train_dir}/knn-{solver_param.N_train}-{n}-{m}-{r}.pickle",
        )

        self._enable_normalize = solver_param.enable_normalize
        if self._enable_normalize:
            mean_std = lambda x: (x.mean(axis=0), x.std(axis=0))
            self.__mean_J, self.__std_J = mean_std(self._J_tr)
            self.__mean_P, self.__std_P = mean_std(self._P_tr)
            self.__mean_F, self.__std_F = mean_std(self._F)

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
        return self._solver_param

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
            file_path = f"{self.param.train_dir}/F-{self.param.N_train}-{self._n}-{self._m}-{self._r}.npy"
            F = load_numpy(file_path=file_path)

            GENERATE_NEW = F.shape != (len(J), self._r)
            if GENERATE_NEW:
                # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
                hnne = HNNE(dim=self._r)
                # maximum number of data for hnne (11M), we use max_num_data_hnne to test
                num_data = min(self.param.max_num_data_hnne, len(J))
                S = np.column_stack((J, P))
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
                                ).flatten()
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
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()
        return torch.from_numpy(((J - self.__mean_J) / self.__std_J).astype(np.float32))

    def norm_C(self, C: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        P = (C[:, : self._m] - self.__mean_P) / self.__std_P
        F = (C[:, self._m : self._m + self._r] - self.__mean_F) / self.__std_F
        noise = C[:, -1]

        return torch.from_numpy(np.column_stack((P, F, noise)).astype(np.float32))

    def denorm_J(self, J: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        device = J.device if isinstance(J, torch.Tensor) else "cpu"
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()
        return torch.from_numpy((J * self.__std_J + self.__mean_J).astype(np.float32))

    def denorm_C(self, C: np.ndarray | torch.Tensor):
        assert self._enable_normalize
        if isinstance(C, torch.Tensor):
            C = C.detach().cpu().numpy()
        P = C[:, : self._m] * self.__std_P + self.__mean_P
        F = C[:, self._m : self._m + self._r] * self.__std_F + self.__mean_F
        noise = C[:, -1]

        return torch.from_numpy(np.column_stack((P, F, noise)).astype(np.float32))

    def sample(self, C, K):
        if self._enable_normalize:
            C = self.norm_C(C)

        C = C.to(self._device)
        with torch.inference_mode():
            J = self._solver(C).sample((K,)).detach().cpu()

        if self._enable_normalize:
            J = self.denorm_J(J)
        return J

    def solve_set_k(
        self,
        single_pose: np.ndarray,
        num_sols: int,
        k: int = 1,
    ):
        """
        _summary_

        Parameters
        ----------
        single_pose : np.ndarray
            a single task point, m=3 or m=7
        num_sols : int
            number of solutions to be sampled from base distribution
        k : int, optional
            number of posture features to be sampled from knn, by default 1
        return_numpy : bool, optional
            return numpy array type or torch cuda tensor type, by default False

        Returns
        -------
        array_like
            array_like(num_sols * k, n_dofs)
        """
        P = (
            single_pose[:, : self._m]
            if len(single_pose.shape) == 2
            else np.atleast_2d(single_pose[: self._m])
        )
        F = nearest_neighbor_F(knn, P, F, n_neighbors=k)  # type: ignore
        P = np.tile(P, (len(F), 1)) if len(P) == 1 and k > 1 else P
        return np.reshape(
            self.solve(P, F, num_sols, return_numpy=True), (num_sols * k, -1)
        )

    def solve(
        self, P: np.ndarray, F: np.ndarray, num_sols: int, return_numpy: bool = False
    ):
        # Begin inference
        J_hat = self.sample(
            torch.from_numpy(
                np.column_stack((P, F, np.zeros((len(F), 1)))).astype(np.float32)
            ),
            num_sols,
        )
        return J_hat.numpy() if return_numpy else J_hat

    def get_random_JPF(self, num_samples: int):
        # Randomly sample poses from train set
        J, P = self._robot.sample_joint_angles_and_poses(
            n=num_samples, return_torch=False
        )
        F = nearest_neighbor_F(self._knn, P, self._F, n_neighbors=1)
        return J, P, F

    def evaluate_solutions(
        self,
        J: np.ndarray | torch.Tensor,
        P: np.ndarray,
        return_row: bool = False,
        return_col: bool = False,
    ):
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
                J_hat=J[:, i, :], P=P[i]
            )  # type: ignore
        if return_row:
            return l2_errs.mean(axis=0), ang_errs.mean(axis=0)
        elif return_col:
            return l2_errs.mean(axis=1), ang_errs.mean(axis=1)
        return l2_errs.mean(), ang_errs.mean()

    def random_sample_solutions_with_evaluation(
        self, num_poses: int, num_sols: int, return_time: bool = False
    ):
        # Randomly sample poses from test set
        P = self._P_ts[np.random.choice(self._P_ts.shape[0], num_poses, replace=False)]
        time_begin = time()
        # Data Preprocessing
        F = nearest_neighbor_F(self._knn, P, self._F)  # type: ignore

        # Begin inference
        J_hat = self.solve(P, F, num_sols, return_numpy=True)
        inference_time = round((time() - time_begin), 3)

        P = P if self._m == 7 else np.column_stack((P, np.ones(shape=(len(P), 4))))

        avg_l2_errs, avg_ang_errs = self.evaluate_solutions(J_hat, P)

        errors_time = round((time() - time_begin), 3) - inference_time
        print(
            tabulate(
                [[inference_time, errors_time]],
                headers=["inference time", "evaluation time"],
            )
        )

        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        if return_time:
            return avg_l2_errs, avg_ang_errs, avg_inference_time
        else:
            return avg_l2_errs, avg_ang_errs

    # def path_following(
    #     self,
    #     load_time: str = "",
    #     num_traj: int = 3,
    #     num_steps=20,
    #     shrink_ratio: float = 0.1,
    #     enable_evaluation: bool = False,
    #     enable_plot: bool = False,
    #     seed: int = 47,
    # ):
    #     """
    #     evaluate the performance of path following

    #     Parameters
    #     ----------
    #     load_time : str, optional
    #         file name of load P_path, by default ""
    #     num_traj : int, optional
    #         number of demo trajectories, by default 3
    #     num_steps : int, optional
    #         length of the generated path, by default 20
    #     shrink_ratio : float, optional
    #         the shrink ratio of the based distribution of IK solver, by default 0.1
    #     enable_evaluation : bool, optional
    #         use evaluation or not, by default False
    #     enable_plot : bool, optional
    #         use plot or not, by default False
    #     """
    #     old_shrink_ratio = self._shrink_ratio
    #     self.shrink_ratio = shrink_ratio

    #     # random sample P_path
    #     P = self.sample_P_path(load_time=load_time, num_steps=num_steps, seed=seed)
    #     P = (
    #         P if self._m == 7 else np.column_stack((P, np.ones((len(P), 4))))
    #     )  # type: ignore

    #     time_begin = time()

    #     # P shape = (num_steps, m)
    #     # Pt shape = (num_traj, num_steps, m)
    #     Pt = np.tile(P, (num_traj, 1, 1)).reshape(-1, self._m)

    #     # get nearest neighbor of p from self.knn
    #     _, idx = self._knn.kneighbors(P, n_neighbors=5)
    #     idx = idx.flatten()
    #     Ft = np.tile(
    #         np.expand_dims(
    #             self._F[idx[np.random.randint(0, len(idx), size=num_traj)]], axis=1
    #         ),
    #         (1, num_steps, 1),
    #     ).reshape(-1, self._r)

    #     Qs = self.solve(Pt, Ft, num_sols=1, return_numpy=True).reshape(
    #         num_traj, num_steps, self._n
    #     )
    #     if enable_evaluation:
    #         mjac_arr = np.array([max_joint_angle_change(qs) for qs in Qs])
    #         # Qs = (num_traj, num_steps, n_dofs)
    #         # evaluate = (num_sols, num_poses, n_dofs)
    #         l2_err_arr, ang_err_arr = self.evaluate_solutions(Qs, P, return_row=True)
    #         df = pd.DataFrame(
    #             {
    #                 "l2_err": l2_err_arr,
    #                 "ang_err": ang_err_arr,
    #                 "mjac": mjac_arr,
    #             }
    #         )
    #         print(df.describe())
    #         print(f"avg_inference_time: {round((time() - time_begin) / num_traj, 3)}")

    #     if enable_plot:
    #         return P, Qs, Ft

    #     self.shrink_ratio = old_shrink_ratio


def max_joint_angle_change(qs: torch.Tensor | np.ndarray):
    if isinstance(qs, torch.Tensor):
        qs = qs.detach().cpu().numpy()
    return np.rad2deg(np.max(np.abs(np.diff(qs, axis=0))))
