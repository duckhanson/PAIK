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
from paik.dataset import data_preprocess_for_inference, nearest_neighbor_F
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional

DEFAULT_SOLVER_PARAM_M3 = SolverConfig(
    lr=0.00033,
    gamma=0.094,
    opt_type="adamw",
    noise_esp=0.001,
    sche_type="steplr",
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
    sche_type="plateau",
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
    sche_type="plateau",
    random_perm=False,
    shrink_ratio=0.68,
    subnet_width=1024,
    num_transforms=8,
    lr_weight_decay=0.012,
    noise_esp_decay=0.97,
    enable_normalize=True,
    subnet_num_layers=3,
    ckpt_name="1124-1758",  # "1123-0919", "1124-1758"
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
            enable_load_model=True,
            num_transforms=solver_param.num_transforms,
            subnet_width=solver_param.subnet_width,
            subnet_num_layers=solver_param.subnet_num_layers,
            shrink_ratio=solver_param.shrink_ratio,
            lr=solver_param.lr,
            lr_weight_decay=solver_param.lr_weight_decay,
            gamma=solver_param.gamma,
            optimizer_type=solver_param.opt_type,
            scheduler_type=solver_param.sche_type,
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
        return_numpy: bool = False,
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
        C = data_preprocess_for_inference(
            P=single_pose, F=self._F, knn=self._knn, m=self._m, k=k
        )

        # Begin inference
        J_hat = torch.reshape(self.sample(C, num_sols), (num_sols * k, -1))

        return J_hat.numpy() if return_numpy else J_hat

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

    def random_sample_JPF(self, num_samples: int):
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
    ) -> tuple[Any, Any]:
        if isinstance(J, torch.Tensor):
            J = J.detach().cpu().numpy()

        num_poses = len(P)
        num_sols = len(J)
        J = np.expand_dims(J, axis=1) if len(J.shape) == 2 else J
        assert J.shape == (num_sols, num_poses, self._robot.n_dofs)

        l2_errs = np.empty((num_poses, num_sols))
        ang_errs = np.empty((num_poses, num_sols))
        for i in range(num_poses):
            l2_errs[i], ang_errs[i] = solution_pose_errors(
                robot=self._robot,
                solutions=J[:, i, :],  # type: ignore
                target_poses=P[i],
                device=self._device,
            )
        if return_row:
            return l2_errs.mean(axis=0), ang_errs.mean(axis=0)
        elif return_col:
            return l2_errs.mean(axis=1), ang_errs.mean(axis=1)
        return l2_errs.mean(), ang_errs.mean()

    def random_evaluation(
        self, num_poses: int, num_sols: int, return_time: bool = False
    ):
        # Randomly sample poses from test set
        P = self._P_ts[np.random.choice(self._P_ts.shape[0], num_poses, replace=False)]
        time_begin = time()
        # Data Preprocessing
        C = data_preprocess_for_inference(P=P, F=self._F, knn=self._knn, m=self._m)

        # Begin inference
        J_hat = self.sample(C, num_sols)
        J_hat = J_hat.detach().cpu().numpy()
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


def solution_pose_errors(
    robot: Robot,
    solutions: np.ndarray,
    target_poses: torch.Tensor | np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the L2 and angular errors of calculated ik solutions for a given target_pose. Note: this function expects
    multiple solutions but only a single target_pose. All of the solutions are assumed to be for the given target_pose

    Args:
        robot (Robot): The Robot which contains the FK function we will use
        solutions (Union[np.ndarray]): [n x 7] IK solutions for the given target pose
        target_pose (np.ndarray): [7] the target pose the IK solutions were generated for

    Returns:
        tuple[np.ndarray, np.ndarray]: The L2, and angular (rad) errors of IK solutions for the given target_pose
    """
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"

    if isinstance(target_poses, torch.Tensor):
        target_poses = target_poses.detach().cpu().numpy()
    target_poses = _get_target_pose_batch(target_poses, solutions.shape[0])

    ee_pose_ikflow = robot.forward_kinematics(solutions[:, 0 : robot.n_dofs])
    rot_output = ee_pose_ikflow[:, 3:]

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_poses[:, 0:3], axis=1)
    rot_target = target_poses[:, 3:]
    assert rot_target.shape == rot_output.shape

    # Surprisingly, this is almost always faster to calculate on the gpu than on the cpu. I would expect the opposite
    # for low number of solutions (< 200).
    q_target_pt = torch.tensor(rot_target, device=device, dtype=torch.float32)
    q_current_pt = torch.tensor(rot_output, device=device, dtype=torch.float32)
    ang_errors = (
        geodesic_distance_between_quaternions(q_target_pt, q_current_pt)
        .detach()
        .cpu()
        .numpy()
    )
    return l2_errors, ang_errors


def max_joint_angle_change(qs: torch.Tensor | np.ndarray):
    if isinstance(qs, torch.Tensor):
        qs = qs.detach().cpu().numpy()
    return np.rad2deg(np.max(np.abs(np.diff(qs, axis=0))))
