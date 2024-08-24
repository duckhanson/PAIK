# Import required packages
from __future__ import annotations
from typing import Any, Tuple
import os
import shutil
from os.path import isdir
from time import time
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from hnne import HNNE
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

from .settings import get_config, SolverConfig, PANDA_PAIK
from .model import get_flow_model, get_robot
from .file import load_numpy, save_numpy, save_pickle, load_pickle
from .evaluate import evaluate_pose_error_P2d_P2d
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional

def get_solver(arch_name: str, robot_name: str, load: bool = False, work_dir: str = os.path.abspath(os.getcwd())) -> Solver:
    """
    Get the solver with the given architecture and robot.

    Args:
        arch_name (str): architecture name
        robot_name (str): robot name
        load (bool, optional): load the solver or not. Defaults to False.

    Returns:
        Solver: solver instance
    """
    solver_param = get_config(arch_name, robot_name)

    if arch_name == "paik":
        solver = PAIK(
            solver_param=solver_param, # type: ignore
            load_date="best" if load else "",
            work_dir=work_dir,
        )
    elif arch_name == "nsf":
        solver = NSF(
            solver_param=solver_param, # type: ignore
            load_date="best" if load else "",
            work_dir=work_dir,
        )
    else:
        raise NotImplementedError(f"Architecture {arch_name} not supported.")
    
    return solver


class Solver:
    def __init__(
        self,
        solver_param: SolverConfig = PANDA_PAIK,
        load_date: str = "",
        work_dir: str = os.path.abspath(os.getcwd()),
    ) -> None:
        solver_param.workdir = work_dir
        self._robot = get_robot(
            solver_param.robot_name, robot_dirs=solver_param.dir_paths
        )

        self.param = solver_param

        if solver_param.use_dimension_reduction:
            print(f"[INFO] use_dimension_reduction is True, use HNNE.")
            raise NotImplementedError("Not support HNNE.")
        else:
            print(f"[INFO] use_dimension_reduction is False, use clustering.")

        try:
            if load_date == "best":
                self.load_best_date()
            else:
                self.load_by_date(load_date)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}. Load training data instead.")
            self._load_training_data()

    @property
    def base_std(self):
        return self._base_std

    @property
    def robot(self):
        return self._robot

    @property
    def param(self):
        return self.__solver_param

    @property
    def latent(self):
        return self._latent

    @param.setter
    def param(self, value: SolverConfig):
        self.__solver_param = value
        self._device = value.device
        self._use_nsf_only = value.use_nsf_only
        # Neural spline flow (NSF) with 3 sample features and 5 context features
        self.n, self.m, self.r = value.n, value.m, value.r
        self._solver, self._optimizer, self._scheduler = get_flow_model(
            value)  # type: ignore
        self._base_std = value.base_std
        self._latent = torch.zeros((self.n,), device=self._device)
        # load inference data
        assert (
            self.n == self.robot.n_dofs
        ), f"n should be {self.robot.n_dofs} as the robot"

    @base_std.setter
    def base_std(self, value: float):
        assert value >= 0, "base_std should be greater than or equal to 0."
        self._base_std = value
        self._change_solver_base()

    @latent.setter
    def latent(self, value: np.ndarray):
        assert len(value) == self.n, f"latent should have length {self.n}."
        self._latent = torch.from_numpy(
            value.astype(np.float32)).to(self._device)
        self._change_solver_base()

    # a dictionary in weight_dir to store the information of top3 dates, their l2, and their model by save_by_date, save the date if the current model is better, and remove the worst date
    def save_if_top3(self, date: str, l2: float):
        top3_date_path = self._top3_date_path()

        if not os.path.exists(top3_date_path):
            save_pickle(
                top3_date_path, {"date": ["", "", ""],
                                 "l2": [1000, 1000, 1000]}
            )
        top3_date = load_pickle(top3_date_path)
        save_idx = -1
        # # if the top3 date has the current date, then check if the current model is better, if so, replace it
        if date in top3_date["date"]:
            if l2 < top3_date["l2"][top3_date["date"].index(date)]:
                save_idx = top3_date["date"].index(date)
        elif l2 < max(top3_date["l2"]):
            save_idx = top3_date["l2"].index(max(top3_date["l2"]))

        if save_idx == -1:
            print(
                f"[INFO] current model is not better than the top3 model in {top3_date_path}"
            )
        else:
            if (
                top3_date["date"][save_idx] != ""
                and top3_date["date"][save_idx] != date
            ):
                self.remove_by_date(top3_date["date"][save_idx])
            top3_date["date"][save_idx] = date
            top3_date["l2"][save_idx] = l2
            save_pickle(top3_date_path, top3_date)
            self.save_by_date(date)
            print(
                f"[SUCCESS] save the date {date} with l2 {l2:.5f} in {top3_date_path}"
            )
        print(
            f"[INFO] top3 dates: {top3_date['date']}, top3 l2: {top3_date['l2']}")

    # remove by date
    def remove_by_date(self, date: str):
        if isdir(os.path.join(self.param.weight_dir, date)):
            shutil.rmtree(os.path.join(
                self.param.weight_dir, date), ignore_errors=True)
            print(f"[SUCCESS] remove {date} in {self.param.weight_dir}.")
        else:
            print(
                f"[WARNING] {date} not found in {self.param.weight_dir}. Remove failed."
            )

    # save model, J, P, F, J_knn, P_knn in the directory of date in the weight_dir
    def save_by_date(self, date: str):
        save_dir = os.path.join(self.param.weight_dir, date)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "solver": self._solver.state_dict(),
            },
            os.path.join(save_dir, "model.pth"),
        )

        # if path exists, do not save again
        if not os.path.exists(os.path.join(save_dir, "J.npy")):
            save_numpy(os.path.join(save_dir, "J.npy"), self.J)
            save_numpy(os.path.join(save_dir, "P.npy"), self.P)
            save_numpy(os.path.join(save_dir, "F.npy"), self.F)

            save_pickle(os.path.join(save_dir, "param.pth"), self.param)
            save_pickle(os.path.join(save_dir, "J_knn.pth"), self.J_knn)
            save_pickle(os.path.join(save_dir, "P_knn.pth"), self.P_knn)
        else:
            print(f"[INFO] J, P, F, J_knn, P_knn already exist in {save_dir}.")
        print(f"[SUCCESS] save model, J, P, F, J_knn, P_knn in {save_dir}")

    def load_by_date(self, date: str):
        if not isdir(os.path.join(self.param.weight_dir, date)):
            raise FileNotFoundError(
                f"{date} not found in {self.param.weight_dir}.")

        load_dir = os.path.join(self.param.weight_dir, date)
        model_path = os.path.join(load_dir, "model.pth")
        param_path = os.path.join(load_dir, "param.pth")
        J_path = os.path.join(load_dir, "J.npy")
        P_path = os.path.join(load_dir, "P.npy")
        F_path = os.path.join(load_dir, "F.npy")
        J_knn_path = os.path.join(load_dir, "J_knn.pth")
        P_knn_path = os.path.join(load_dir, "P_knn.pth")

        self.param = load_pickle(param_path)
        self._solver.load_state_dict(torch.load(model_path)["solver"])
        self.J = np.load(J_path)
        self.P = np.load(P_path)
        if self._use_nsf_only:
            self.C = self.P
        else:
            self.F = np.load(F_path)
            self.C = np.column_stack((self.P, self.F))
        self._compute_normalizing_elements()
        self.J_knn = load_pickle(J_knn_path)
        self.P_knn = load_pickle(P_knn_path)

        print(f"[SUCCESS] load from {load_dir}")

    def _top3_date_path(self):
        if self._use_nsf_only:
            return os.path.join(self.param.weight_dir, "top3_date_nsf.pth")
        else:
            return os.path.join(self.param.weight_dir, "top3_date_paik.pth")

    def load_best_date(self):
        top3_date_path = self._top3_date_path()

        if not os.path.exists(top3_date_path):
            raise FileNotFoundError(
                f"{top3_date_path} not found. Please save the model first."
            )
        top3_date = load_pickle(top3_date_path)

        best_date = top3_date["date"][top3_date["l2"].index(
            min(top3_date["l2"]))]
        self.load_by_date(best_date)
        print(
            f"[SUCCESS] load best date {best_date} with l2 {min(top3_date['l2']):.5f} from {top3_date_path}."
        )

    # private methods
    def _compute_normalizing_elements(self):
        self._normalization_elements = {
            "J": {"mean": self.J.mean(axis=0), "std": self.J.std(axis=0)},
            "C": {
                # extra column for tuning std of noise for training data
                "mean": np.concatenate((self.C.mean(axis=0), np.zeros((1)))),
                "std": np.concatenate((self.C.std(axis=0), np.ones((1)))) + 1e-6,
            },
        }
        
    def _load_J_P(self):
        """
        Load J and P from the given path, if not found, generate and save it.
        """
        J, P = [load_numpy(file_path=self._load_JPF_path(name))
                   for name in ["J", "P"]]

        if J is None or P is None:
            print(
                f"[WARNING] J or P not found, generate and save in {self._load_JPF_path('J')}.")
            J, P = self.robot.sample_joint_angles_and_poses(
                n=self.param.N
            )
            save_numpy(file_path=self._load_JPF_path("J"), arr=J)
            save_numpy(file_path=self._load_JPF_path("P"), arr=P)
            print(
                f"[SUCCESS] J and P saved in {self._load_JPF_path('J')} and {self._load_JPF_path('P')}.")

        self.J, self.P = J, P
        
    def _load_JPF_path(self, name: str):
        """
        JPF path for saving and loading J, P, F.
        """
        assert isdir(
            self.param.train_dir
        ), f"{self.param.train_dir} not found, please change workdir to the project root!"
        return f"{self.param.train_dir}/{name}-{self.param.N}-{self.n}-{self.m}-{self.r}.npy"
        
        
    def _load_F(self):
        """
        Load F from the given path, if not found, generate and save it.
        """
        F = load_numpy(file_path=self._load_JPF_path("F"))

        if F is None:
            print(
                f"[WARNING] F not found, generate and save in {self._load_JPF_path('F')}.")
            assert self.r > 0, "r should be greater than 0."
            # maximum number of data for hnne (11M), we use max_num_data_hnne to test
            num_data = min(self.param.max_num_data_hnne, len(J))

            # hnne = HNNE(dim=r, ann_threshold=config.num_neighbors)
            hnne = HNNE()
            F = hnne.fit_transform(X=J[:num_data], dim=self.r, verbose=True)

            if not self.param.use_dimension_reduction:
                print(f"[INFO] use_dimension_reduction is False, use clustering.")
                partitions = hnne.hierarchy_parameters.partitions
                num_clusters = hnne.hierarchy_parameters.partition_sizes
                closest_idx_to_num_clusters_20 = np.argmin(
                    np.abs(np.array(num_clusters) - 20)
                )
                F = partitions[:,
                               closest_idx_to_num_clusters_20].reshape(-1, 1)

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
                )

            save_numpy(file_path=self._load_JPF_path("F"), arr=F)
            print(f"[SUCCESS] F saved in {self._load_JPF_path('F')}.")

        self.F = F
        
        if not self.param.use_dimension_reduction:
            # check if numbers of F are integers
            assert np.allclose(self.F, self.F.astype(int)), "F should be integers."
        

    def _load_training_data(self):
        """
        Load training data from the given path, if not found, generate and save it.
        """
        self._load_J_P()
        self._load_F()

        self.C = np.column_stack((self.P, self.F))
        
        # for normalization
        self._compute_normalizing_elements()
        self._load_knn()

    def _load_knn(self):
        """
        Load knn models from the given path, if not found, generate and save it.
        """
        
        path_P_knn = f"{self.param.weight_dir}/P_knn-{self.param.N}-{self.n}-{self.m}-{self.r}.pth"
        try:
            self.P_knn = load_pickle(path_P_knn)
        except:
            print(
                f"[WARNING] P_knn not found, generate and save in {path_P_knn}.")
            self.P_knn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(self.P)
            save_pickle(
                path_P_knn,
                self.P_knn,
            )
        print(f"[SUCCESS] P_knn load from {path_P_knn}.")

        path_J_knn = f"{self.param.weight_dir}/J_knn-{self.param.N}-{self.n}-{self.m}-{self.r}.pth"
        try:
            self.J_knn = load_pickle(path_J_knn)
        except:
            print(
                f"[WARNING] J_knn not found, generate and save in {path_J_knn}.")
            self.J_knn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(self.J)
            save_pickle(
                path_J_knn,
                self.J_knn,
            )
        print(f"[SUCCESS] J_knn load from {path_J_knn}.")

    def _change_solver_base(self):
        """
        Change the base distribution of the solver to the new base distribution.
        """
        self._solver = Flow(
            transforms=self._solver.transforms,  # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self.robot.n_dofs,),
                            device=self._device) + self._latent,
                torch.ones((self.robot.n_dofs,),
                           device=self._device) * self._base_std,
                buffer=True,
            ),  # type: ignore
        )

    # public methods
    def normalize_input_data(self, data: np.ndarray, name: str, return_torch: bool = False):
        """
        Normalize input data. J_norm = (J - J_mean) / J_std, C_norm = (C - C_mean) / C_std

        Args:
            data (np.ndarray): input only support J and C
            name (str): name of the input data, only support "J" and "C"

        Returns:
            np.ndarray: normalized data
        """
        norm = (data - self._normalization_elements[name]["mean"]) / self._normalization_elements[name]["std"]
        if return_torch:
            return torch.from_numpy(
                norm.astype(np.float32)
            ).to(self._device)
        
        return norm

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
            data * self._normalization_elements[name]["std"]
            + self._normalization_elements[name]["mean"]
        )

    def _remove_partition_label(self, C: np.ndarray):
        """
        Remove posture feature from C

        Args:
            C (np.ndarray): conditions

        Returns:
            np.ndarray: conditions without posture feature
        """
        assert self._use_nsf_only and isinstance(C, np.ndarray)
        # delete the last 2 of the last dimension of C, which is posture feature
        C = np.delete(C, -2, -1)
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
        C = self._remove_partition_label(C) if self._use_nsf_only else C
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
        C = np.concatenate((C, C[:complementary]),
                           axis=0) if complementary > 0 else C
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
    
    def _get_batch_C(self, P: np.ndarray, F: np.ndarray, num_sols: int, batch_size: int):
        # shape: (num_poses, C.shape[-1] = m + r + 1)
        C = np.column_stack((P, F, np.zeros((len(F), 1))))
        C = self.normalize_input_data(C, "C")
        # C: (num_poses, m + r + 1) -> C: (num_sols * num_poses, m + r + 1)
        C = np.tile(C, (num_sols, 1))
        C, complementary = self.make_divisible_C(C, batch_size)
        C = torch.from_numpy(
            C.astype(np.float32).reshape(-1, batch_size, C.shape[-1])
        ).to(self._device)  
        return C, complementary
    
    def _solve_C_batch(self, C: np.ndarray, num_sols: int, complementary: int, verbose: bool=False) -> np.ndarray:
        """
        Solve inverse kinematics problem in batch.

        Args:
            C (np.ndarray): conditions with shape (num_poses, m + x + 1), x = 1 for paik, x = 0 for nsf

        Returns:
            np.ndarray: J with shape (num_poses, num_dofs)
        """
        batch_size = C.shape[1]
        J = torch.empty((len(C), batch_size, self._robot.n_dofs),
                        device=self._device)
        
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

        C, complementary = self._get_batch_C(P, F, num_sols, batch_size)
        J = self._solve_C_batch(C, num_sols, complementary, verbose)
        
        return J

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
        assert len(J.shape) == 3 and len(
            P.shape) == 2 and J.shape[1] == num_poses, f"J: {J.shape}, P: {P.shape}"

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

    def get_reference_partition_label(
        self, P: np.ndarray, select_reference: str = "knn", num_sols: int = 1
    ):
        if select_reference == "zero":
            return np.zeros((len(P), num_sols)).flatten()
        elif select_reference == "knn":
            # type: ignore
            n_neighbors = min(num_sols, 10)
            F = self.F[
                self.P_knn.kneighbors(
                    np.atleast_2d(P), n_neighbors=n_neighbors, return_distance=False
                )
            ].reshape(-1, n_neighbors)
            # expand F to match the number of solutions by random sampling for each pose
            return np.asarray([np.random.choice(f, num_sols, replace=True) for f in np.atleast_2d(F)]).flatten()
        elif select_reference == "random":
            min_F, max_F = np.min(self.F), np.max(self.F)
            return np.random.rand(len(P), self.r) * (max_F - min_F) + min_F
        elif select_reference == "pick":
            # randomly pick one posture from train set
            return self.F[np.random.randint(0, len(self.F), len(P))]
        else:
            raise NotImplementedError

    def generate_ik_solutions(
        self,
        P: np.ndarray,
        F: np.ndarray,
        num_sols: int,
        std: float,
        latent: np.ndarray,
    ):
        assert len(P) == len(F), "P and F should have the same length."
        if std != self.base_std:
            self.base_std = std
        if not np.array_equal(latent, np.zeros((self.n,))):
            self.latent = latent
        J_hat = self.solve_batch(P, F, num_sols)
        return J_hat

    def evaluate_ikp_iterative(
        self,
        num_poses: int,
        num_sols: int,
        batch_size: int = 5000,
        std: float = 0.25,
        success_threshold: Tuple[float, float] = (1e-4, 1e-4),
        select_reference: str = "knn",
        verbose: bool = True,
    ):  # -> tuple[Any, Any, float] | tuple[Any, Any]:# -> tuple[Any, Any, float] | tuple[Any, Any]:
        self.base_std = std
        # Randomly sample poses from test set
        _, P = self.robot.sample_joint_angles_and_poses(n=num_poses)
        time_begin = time()
        # Data Preprocessing
        F = self.get_reference_partition_label(P, select_reference)

        # Begin inference
        J_hat = self.solve_batch(
            P, F, num_sols, batch_size=batch_size, verbose=verbose)

        l2, ang = self.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        df = pd.DataFrame({"l2": l2, "ang": np.rad2deg(ang)})
        print(df.describe())

        print(
            tabulate(
                [
                    [
                        np.round(l2.mean() * 1e3, decimals=2),
                        np.round(np.rad2deg(ang.mean()), decimals=2),
                        np.round(avg_inference_time * 1e3, decimals=0),
                    ]
                ],
                headers=[
                    "l2 (mm)",
                    "ang (deg)",
                    "inference_time (ms)",
                ],
            )
        )

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


class PAIK(Solver):
    def __init__(
        self,
        solver_param: SolverConfig = PANDA_PAIK,
        load_date: str = "",
        work_dir: str = os.path.abspath(os.getcwd()),
    ) -> None:
        super().__init__(solver_param, load_date, work_dir)
        
class NSF(Solver):
    def __init__(
        self,
        solver_param: SolverConfig = PANDA_PAIK,
        load_date: str = "",
        work_dir: str = os.path.abspath(os.getcwd()),
    ) -> None:
        super().__init__(solver_param, load_date, work_dir)
        
    def _load_training_data(self):
        """
        Load training data from the given path, if not found, generate and save it.
        """
        self._load_J_P()

        self.C = self.P
        
        # for normalization
        self._compute_normalizing_elements()
        self._load_knn()
        
        
    def solve(self, P: np.ndarray, num_sols: int):
        C = self.normalize_input_data(np.column_stack((P, np.zeros((len(P), 1)))), "C", return_torch=True)
        with torch.inference_mode():
            J = self._solver(C).sample((num_sols,))
        return self.denormalize_output_data(J.detach().cpu().numpy(), "J")
    
    def _get_batch_C(self, P: np.ndarray, num_sols: int, batch_size: int):
        # shape: (num_poses, C.shape[-1] = m + r + 1)
        C = np.column_stack((P, np.zeros((len(P), 1))))
        C = self.normalize_input_data(C, "C")
        # C: (num_poses, m + r + 1) -> C: (num_sols * num_poses, m + r + 1)
        C = np.tile(C, (num_sols, 1))
        C, complementary = self.make_divisible_C(C, batch_size)
        C = torch.from_numpy(
            C.astype(np.float32).reshape(-1, batch_size, C.shape[-1])
        ).to(self._device)  
        return C, complementary

    def solve_batch(
        self,
        P: np.ndarray,
        num_sols: int,
        batch_size: int = 4000,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Solve inverse kinematics problem in batch.

        Args:
            P (np.ndarray): poses with shape (num_poses, m)
            num_sols (int): number of solutions
            batch_size (int, optional): batch size. Defaults to 4000.
            verbose (bool, optional): use trange or not. Defaults to True.

        Returns:
            np.ndarray: J with shape (num_sols, num_poses, num_dofs)
        """

        if len(P) * num_sols < batch_size:
            return self.solve(P, num_sols)

        C, complementary = self._get_batch_C(P, num_sols, batch_size)
        J = self._solve_C_batch(C, num_sols, complementary, verbose)
        return J
    
    def evaluate_ikp_iterative(
        self,
        num_poses: int,
        num_sols: int,
        batch_size: int = 5000,
        std: float = 0.25,
        success_threshold: Tuple[float, float] = (1e-4, 1e-4),
        select_reference: str = "knn",
        verbose: bool = True,
    ):  # -> tuple[Any, Any, float] | tuple[Any, Any]:# -> tuple[Any, Any, float] | tuple[Any, Any]:
        self.base_std = std
        # Randomly sample poses from test set
        _, P = self.robot.sample_joint_angles_and_poses(n=num_poses)
        time_begin = time()

        # Begin inference
        J_hat = self.solve_batch(P, num_sols, batch_size=batch_size, verbose=verbose)

        l2, ang = self.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        df = pd.DataFrame({"l2": l2, "ang": np.rad2deg(ang)})
        print(df.describe())

        print(
            tabulate(
                [
                    [
                        np.round(l2.mean() * 1e3, decimals=2),
                        np.round(np.rad2deg(ang.mean()), decimals=2),
                        np.round(avg_inference_time * 1e3, decimals=0),
                    ]
                ],
                headers=[
                    "l2 (mm)",
                    "ang (deg)",
                    "inference_time (ms)",
                ],
            )
        )

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
        
