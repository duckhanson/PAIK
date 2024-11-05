# Import required packages
from __future__ import annotations
from typing import Any, Optional, Tuple
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
from .model import get_flow_model, get_robot, Flow
from .file import load_numpy, save_numpy, save_pickle, load_pickle
from .evaluate import evaluate_pose_error_P2d_P2d
from zuko.distributions import DiagNormal, BoxUniform
from zuko.flows import Unconditional


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
            solver_param=solver_param,  # type: ignore
            load_date="best" if load else "",
            work_dir=work_dir,
        )
    elif arch_name == "nsf":
        solver = NSF(
            solver_param=solver_param,  # type: ignore
            load_date="best" if load else "",
            work_dir=work_dir,
        )
    else:
        raise NotImplementedError(f"Architecture {arch_name} not supported.")

    return solver


def get_solver_from_config(solver_param: SolverConfig, load: bool = False) -> Solver:
    """
    Get the solver with the given solver config.

    Args:
        solver_param (SolverConfig): solver config
        load (bool, optional): load the solver or not. Defaults to False.

    Returns:
        Solver: solver instance
    """
    if solver_param.model_architecture == "paik":
        solver = PAIK(
            solver_param=solver_param,  # type: ignore
            load_date="best" if load else "",
            work_dir=solver_param.workdir,
        )
    elif solver_param.model_architecture == "nsf":
        solver = NSF(
            solver_param=solver_param,  # type: ignore
            load_date="best" if load else "",
            work_dir=solver_param.workdir,
        )
    else:
        raise NotImplementedError(
            f"Architecture {solver_param.model_architecture} not supported.")

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

        # if solver_param.use_dimension_reduction:
        #     print(f"[INFO] use_dimension_reduction is True, use HNNE.")
        #     raise NotImplementedError("Not support HNNE.")
        # else:
        #     print(f"[INFO] use_dimension_reduction is False, use clustering.")

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
    
    @property
    def base_name(self):
        return self._base_name

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
        self._base_name = "diag_normal"
        # load inference data
        assert (
            self.n == self.robot.n_dofs
        ), f"n should be {self.robot.n_dofs} as the robot"

    @base_std.setter
    def base_std(self, value: float):
        assert value >= 0, "base_std should be greater than or equal to 0."
        self._base_std = value
        self._update_base_distribution()

    @latent.setter
    def latent(self, value: np.ndarray):
        assert len(value) == self.n, f"latent should have length {self.n}."
        self._latent = torch.from_numpy(
            value.astype(np.float32)).to(self._device)
        self._update_base_distribution()
        
    @base_name.setter
    def base_name(self, value: str):
        assert value in ["diag_normal", "box_uniform"], "base_name should be in ['diag_normal', 'box_uniform']."
        self._base_name = value
        self._update_base_distribution()

    # a dictionary in weight_dir to store the information of best date, their l2, and their model by save_by_date, save the date if the current model is better, and remove the worst date
    def save_if_best(self, date: str, l2: float):
        best_date_path = self._best_date_path()

        if not os.path.exists(best_date_path):
            df = pd.DataFrame({"date": ["first_init"], "l2": [1000]})
            df.to_csv(best_date_path, index=False)
            print(f"[INFO] create {best_date_path} with first_init.")
            
        df = pd.read_csv(best_date_path)
        best_date = df["date"].values[0]
        best_l2 = df["l2"].values[0]
        
        if l2 < best_l2:
            self._remove_by_date(best_date)
            best_date = date
            best_l2 = l2
            df = pd.DataFrame({"date": [best_date], "l2": [best_l2]})
            df.to_csv(best_date_path, index=False)
            self._save_by_date(date)
            print(
                f"[SUCCESS] save the date {date} with l2 {l2:.5f} in {best_date_path}"
            )
        else:
            print(
                f"[INFO] current model is not better than the best model in {best_date_path}"
            )
            
        print(
            f"[INFO] best date: {best_date}, best l2: {best_l2}")

    # remove by date
    def _remove_by_date(self, date: str):
        if date == "":
            print(f"[WARNING] date is empty. Remove failed.")
            return
        
        if isdir(os.path.join(self.param.weight_dir, date)):
            shutil.rmtree(os.path.join(
                self.param.weight_dir, date), ignore_errors=True)
            print(f"[SUCCESS] remove {date} in {self.param.weight_dir}.")
        else:
            print(
                f"[WARNING] {date} not found in {self.param.weight_dir}. Remove failed."
            )

    # save model, J, P, F, J_knn, P_knn in the directory of date in the weight_dir
    def _save_by_date(self, date: str):
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
            if not self._use_nsf_only:
                save_numpy(os.path.join(save_dir, "F.npy"), self.F)

            save_pickle(os.path.join(save_dir, "param.pth"), self.param)
            save_pickle(os.path.join(save_dir, "J_knn.pth"), self.J_knn)
            save_pickle(os.path.join(save_dir, "P_knn.pth"), self.P_knn)
        else:
            print(f"[INFO] J, P, F, J_knn, P_knn already exist in {save_dir}.")
        print(f"[SUCCESS] save model, J, P, F, J_knn, P_knn in {save_dir}")

    def _load_by_date(self, date: str):
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
        try:
            self._solver.load_state_dict(torch.load(model_path)["solver"])
        except RuntimeError as e:
            print(f"[Warning] {e}. Please check the model path {model_path}.")

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

    def _best_date_path(self):
        if self._use_nsf_only:
            return os.path.join(self.param.weight_dir, "best_date_nsf.csv")
        else:
            return os.path.join(self.param.weight_dir, "best_date_paik.csv")

    def _load_best_date(self):
        best_date_path = self._best_date_path()
        
        if not os.path.exists(best_date_path):
            raise FileNotFoundError(
                f"{best_date_path} not found. Please save the model first."
            )
            
        # read the best date from the csv file
        df = pd.read_csv(best_date_path)
        best_date = df["date"].values[0]
        best_l2 = df["l2"].values[0]
        self._load_by_date(best_date)

        print(
            f"[SUCCESS] load best date {best_date} with l2 {best_l2:.5f} from {best_date_path}."
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
                # print(f"[INFO] use_dimension_reduction is False, use clustering.")
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
            assert np.allclose(self.F, self.F.astype(int)
                               ), "F should be integers."

    def _load_training_data(self):
        """
        Load training data from the given path, if not found, generate and save it.
        """
        self._load_J_P()
        self._load_F()

        self.C = np.column_stack((self.P, self.F))

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

    def _update_diagNormal_base(self):
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
    
    def _update_boxUniform_base(self):
        """
        Change the base distribution of the solver to the new base distribution.
        """
        bound = torch.ones((self.robot.n_dofs,),
                           device=self._device) * self._base_std
                
        self._solver = Flow(
            transforms=self._solver.transforms,  # type: ignore
            base=Unconditional(
                BoxUniform,
                -1 * bound,
                bound,
                buffer=True,
            ),  # type: ignore
        )
    
    def _update_base_distribution(self):
        """
        Change the base distribution of the solver to the new base distribution.
        
        Args:
            base_name (str): name of the new base distribution, only support "diag_normal" and "box_uniform"
            std (float): standard deviation of the new base distribution
        """
        if self.base_name == "diag_normal":
            self._update_diagNormal_base()
        else:
            self._update_boxUniform_base()
        
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
        norm = (data - self._normalization_elements[name]
                ["mean"]) / self._normalization_elements[name]["std"]
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

    def _solve_conditions(self, conditions: np.ndarray, num_sols: int):
        conditions_torch = self.normalize_input_data(
            conditions, "C", return_torch=True)
        with torch.inference_mode():
            J = self._solver(conditions_torch).sample((num_sols,))
        return self.denormalize_output_data(J.detach().cpu().numpy(), "J")

    def _solve_conditions_z(self, conditions: np.ndarray, z: np.ndarray):
        conditions_torch = self.normalize_input_data(
            conditions, "C", return_torch=True)
        z_torch = torch.from_numpy(z.astype(np.float32)).to(self._device)
        with torch.inference_mode():
            J = self._solver(conditions_torch).sample_x_from_z(z_torch)
        return self.denormalize_output_data(J.detach().cpu().numpy(), "J")

    def _get_divisible_conditions(
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

    def _remove_complementary_ik_solutions(self, J: np.ndarray, complementary: int) -> np.ndarray:
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

    def _get_conditions(self, P: np.ndarray, F: Optional[np.ndarray] = None):
        if F is None:
            return np.column_stack((P, np.zeros((len(P), 1))))
        else:
            return np.column_stack((P, F, np.zeros((len(F), 1))))

    def _get_conditions_batch(self, P: np.ndarray, num_sols: int, batch_size: int, F: Optional[np.ndarray] = None):
        # shape: (num_poses, C.shape[-1] = m + r + 1)
        C = self._get_conditions(P, F)
        C = self.normalize_input_data(C, "C")
        # C: (num_poses, m + r + 1) -> C: (num_sols * num_poses, m + r + 1)
        C = np.tile(C, (num_sols, 1))
        C, complementary = self._get_divisible_conditions(C, batch_size)
        C = torch.from_numpy(
            C.astype(np.float32).reshape(-1, batch_size, C.shape[-1])
        ).to(self._device)
        return C, complementary
    
    def _solve_conditions_batch(self, C: np.ndarray, num_sols: int, complementary: int, verbose: bool = False) -> np.ndarray:
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
        J = self._remove_complementary_ik_solutions(
            J.reshape(-1, self._robot.n_dofs), complementary
        )
        return self.denormalize_output_data(
            J.reshape(num_sols, -1, self._robot.n_dofs), "J"
        )
        
    def _solve_conditions_batch_z(self, C: torch.Tensor, Z: torch.Tensor, num_sols: int, complementary: int, verbose: bool = False) -> np.ndarray:
        """
        Solve inverse kinematics problem in batch.

        Args:
            C (torch.Tensor): conditions with shape (num_poses, m + x + 1), x = 1 for paik, x = 0 for nsf
            Z (torch.Tensor): latent variables with shape (num_sols, num_poses, z)

        Returns:
            np.ndarray: J with shape (num_poses, num_dofs)
        """
        batch_size = C.shape[1]
        J = torch.empty((len(C), batch_size, self._robot.n_dofs),
                        device=self._device)

        iterator = trange(len(C)) if verbose else range(len(C))
        with torch.inference_mode():
            for i in iterator:
                J[i] = self._solver(C[i]).sample_x_from_z(Z[i])

        J = J.detach().cpu().numpy()
        J = self._remove_complementary_ik_solutions(
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

    def generate_ik_solutions(self, *args, **kwargs):
        raise NotImplementedError("generate_ik_solutions not implemented.")
    
    def generate_ik_solutions_z(self, *args, **kwargs):
        raise NotImplementedError("generate_ik_solutions_z not implemented.")
    
    def generate_z_from_ik_solutions(self, *args, **kwargs):
        raise NotImplementedError("generate_z_from_ik_solutions not implemented.")

    def generate_z_from_dataset(self, *args, **kwargs):
        raise NotImplementedError("generate_z_from_dataset not implemented.")

    def random_ikp(
        self,
        num_poses: int,
        num_sols: int,
        batch_size: int = 5000,
        std: float = 0.25,
        # success_threshold: Tuple[float, float] = (1e-4, 1e-4),
        select_reference: str = "knn",
        verbose: bool = True,
    ):  # -> tuple[Any, Any, float] | tuple[Any, Any]:# -> tuple[Any, Any, float] | tuple[Any, Any]:
        self.base_std = std
        # Randomly sample poses from test set
        _, P = self.robot.sample_joint_angles_and_poses(n=num_poses)
        time_begin = time()

        J_hat = self.generate_ik_solutions(
            P=P, num_sols=num_sols, batch_size=batch_size, verbose=verbose)

        l2, ang = self.evaluate_pose_error_J3d_P2d(J_hat, P, return_all=True)
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        df = pd.DataFrame({"l2": l2, "ang": np.rad2deg(ang)})

        if verbose:
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
                # round(
                #     len(
                #         df.query(
                #             f"l2 < {success_threshold[0]} & ang < {success_threshold[1]}"
                #         )
                #     )
                #     / (num_poses * num_sols),
                #     3,
                # ),
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

        try:
            if load_date == "best":
                self._load_best_date()
            else:
                self._load_by_date(load_date)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}. Load training data instead.")
            self._load_training_data()

    def get_reference_partition_label(
        self, P: np.ndarray, select_reference: str = "knn", num_sols: int = 1
    ):
        if select_reference == "knn":
            # type: ignore
            n_neighbors = min(num_sols, 30)
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
            raise NotImplementedError(
                f"select_reference {select_reference} not supported.")

    def generate_ik_solutions(
        self,
        P: np.ndarray,
        num_sols: int = 1,
        F: Optional[np.ndarray] = None,
        std: Optional[float] = None,
        latent: Optional[np.ndarray] = None,
        select_reference: str = "knn",
        batch_size: int = 4000,
        verbose: bool = True,
    ):
        # shape: (num_sols, num_poses, m)
        P_num_sols = np.expand_dims(P, axis=0).repeat(num_sols, axis=0)
        # shape: (num_sols*num_poses, n)
        P_num_sols = P_num_sols.reshape(-1, P.shape[-1])
        
        if F is None:
            F = self.get_reference_partition_label(P, select_reference, num_sols)

        assert len(P_num_sols) == len(F), "P and F should have the same length."
        if std is not None and std != self.base_std:
            self.base_std = std
        if latent is not None:
            self.latent = latent

        if len(P_num_sols) < batch_size:
            conditions = self._get_conditions(P_num_sols, F)
            return self._solve_conditions(conditions, 1)

        C, complementary = self._get_conditions_batch(
            P=P_num_sols, num_sols=1, batch_size=batch_size, F=F)
        return self._solve_conditions_batch(C, 1, complementary, verbose)


class NSF(Solver):
    def __init__(
        self,
        solver_param: SolverConfig = PANDA_PAIK,
        load_date: str = "",
        work_dir: str = os.path.abspath(os.getcwd()),
    ) -> None:
        super().__init__(solver_param, load_date, work_dir)

        try:
            if load_date == "best":
                self._load_best_date()
            else:
                self._load_by_date(load_date)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}. Load training data instead.")
            self._load_training_data()
        
        latent_path = os.path.join(self.param.weight_dir, "Z.npy")
        if os.path.exists(latent_path):
            self.Z = load_numpy(latent_path)
            print(f"[INFO] Load latent variable from {latent_path}.")
        else:
            self.Z = self.generate_z_from_dataset()
            # save the latent variable
            save_numpy(os.path.join(self.param.weight_dir, "Z.npy"), self.Z)
            print(f"[SUCCESS] save latent variable in {self.param.weight_dir}/Z.npy.")

    def _load_training_data(self):
        """
        Load training data from the given path, if not found, generate and save it.
        """
        self._load_J_P()

        self.C = self.P

        self._compute_normalizing_elements()
        self._load_knn()

    # def generate_ik_solutions(
    #     self,
    #     P: np.ndarray,
    #     num_sols: int = 1,
    #     std: Optional[float] = None,
    #     latent: Optional[np.ndarray] = None,
    #     batch_size: int = 4000,
    #     verbose: bool = True,
    # ):
    #     if std is not None and std != self.base_std:
    #         self.base_std = std
    #     if latent is not None:
    #         self.latent = latent

    #     if len(P) * num_sols < batch_size:
    #         C = self._get_conditions(P)
    #         return self._solve_conditions(C, num_sols)

    #     C, complementary = self._get_conditions_batch(
    #         P=P, num_sols=num_sols, batch_size=batch_size)
    #     return self._solve_conditions_batch(C, num_sols, complementary, verbose)

    def generate_ik_solutions_z(
        self,
        P: np.ndarray,
        z: np.ndarray,
    ):
        """
        Generate inverse kinematics solutions given the conditions and latent variable

        example:
        num_poses = 10
        num_sols = 1000
        Q, P = nsf.robot.sample_joint_angles_and_poses(n=num_poses)
        conditions_torch = c.to('cuda')
        conditions_torch = conditions_torch.to(torch.float32)
        z = np.random.randn(num_sols, num_poses, nsf.n) * 0.25
        J_hat = nsf.generate_ik_solutions_z(P, z)
        
        Args:
            P (np.ndarray): conditions of EE poses with shape (num_poses, m)
            z (np.ndarray): latent variable with shape (num_sols, n)

        Returns:
            np.ndarray: joint angles with shape (num_sols, num_poses, num_dofs)
        """
        C = self._get_conditions(P)
        return self._solve_conditions_z(C, z)
    
    def generate_z_from_ik_solutions(
        self,
        P: np.ndarray,
        J: np.ndarray,
    ):
        """
        Generate latent variable given the conditions and joint angles

        example:
        num_poses = 10
        num_sols = 1000
        Q, P = nsf.robot.sample_joint_angles_and_poses(n=num_poses)
        conditions_torch = c.to('cuda')
        conditions_torch = conditions_torch.to(torch.float32)
        J_hat = nsf.generate_ik_solutions(P, num_sols)
        z_hat = nsf.generate_z_from_ik_solutions(P, J_hat)
        
        Args:
            P (np.ndarray): conditions of EE poses with shape (num_poses, m)
            J (np.ndarray): joint angles with shape (num_sols, num_poses, num_dofs)

        Returns:
            np.ndarray: latent variable with shape (num_sols, num_poses, n)
        """
        C = self._get_conditions(P)
        C = self.normalize_input_data(C, "C", return_torch=True)
        J = self.normalize_input_data(J, "J", return_torch=True)
        with torch.inference_mode():
            z = self._solver(C).sample_z_from_x(J)
        return z.detach().cpu().numpy()
    
    def generate_z_from_dataset(
        self
    ):
        """
        Generate latent variable from the dataset.

        Returns:
            np.ndarray: latent variable with shape (num_data, n)
        """
        print("[INFO] Generate latent variable from the dataset.")
        
        C = self._get_conditions(self.P)
        
        batch_size = 4000
        C_batch, complementary = self._get_divisible_conditions(C, batch_size)
        J_batch, _ = self._get_divisible_conditions(self.J, batch_size)
        C_batch = C_batch.reshape(-1, batch_size, C_batch.shape[-1])
        J_batch = J_batch.reshape(-1, batch_size, J_batch.shape[-1])
        C_batch = self.normalize_input_data(C_batch, "C", return_torch=True)
        J_batch = self.normalize_input_data(J_batch, "J", return_torch=True)
        
        Z = torch.empty_like(J_batch)
        
        with torch.inference_mode():
            for i in trange(len(C_batch)):
                Z[i] = self._solver(C_batch[i]).sample_z_from_x(J_batch[i])
        Z = Z.detach().cpu().numpy()
        Z = self._remove_complementary_ik_solutions(
            Z.reshape(-1, self._robot.n_dofs), complementary
        )
        return Z
    
    def _get_latent_batch(self, Z: np.ndarray, batch_size: int):
        Z, complementary = self._get_divisible_conditions(Z, batch_size)
        Z = torch.from_numpy(
            Z.astype(np.float32).reshape(-1, batch_size, Z.shape[-1])
        ).to(self._device)
        return Z, complementary
    
    def generate_ik_solutions(
        self,
        P: np.ndarray,
        num_sols: int = 1,
        std: Optional[float] = None,
        latent: Optional[np.ndarray] = None,
        random_sample_size: int = 10,
        batch_size: int = 4000,
        verbose: bool = True,
    ):
        if std is not None and std != self.base_std:
            self.base_std = std

        if latent is None:
            # print("[INFO] generate latent variable from retrieval of Random J.")
            J_random = self.robot.sample_joint_angles(n=len(P)*num_sols)
            latent = self.retrieve_latent(J=J_random, random_sample_size=random_sample_size)
            latent = latent.reshape(-1, self.n)
        
        C = self._get_conditions(P)
        
        if len(P) * num_sols < batch_size:
            return self._solve_conditions_z(C, latent)
        else:
            C_batch, complementary = self._get_conditions_batch(
                P=P, num_sols=num_sols, batch_size=batch_size)
            
            Z_batch, _ = self._get_latent_batch(latent, batch_size)
            
            return self._solve_conditions_batch_z(C_batch, Z_batch, num_sols, complementary, verbose)

        return self._solve_conditions_batch_z(C_batch, latent_batch, 1, complementary, verbose)
    
    def retrieve_latent(
        self,
        P: Optional[np.ndarray] = None,
        J: Optional[np.ndarray] = None,
        random_sample_size: int = 0,
    ):
        """
        Retrieve latent variable from the dataset given the conditions and joint angles.

        Args:
            P (np.ndarray, optional): conditions of EE poses with shape (num_poses, m). Defaults to None.
            J (np.ndarray, optional): joint angles with shape (num_sols, num_poses, num_dofs). Defaults to None.
            k (int, optional): number of nearest neighbors. Defaults to 1.
        Returns:
            np.ndarray: latent variable with shape (k, num_poses, n)
        """
        if P is not None:
            num_poses = len(P)
        elif J is not None:
            num_poses = len(J)
        else:
            raise ValueError("P or J should be provided.")
        
        if random_sample_size == 0:
            if P is not None:
                ids = self.P_knn.kneighbors(P, return_distance=False)
            else:
                ids = self.J_knn.kneighbors(J, return_distance=False)
            # shape: (num_poses, k) -> shape: (k, num_poses)  -> shape: (k * num_poses)   
            ids = ids.T.flatten() 
        elif random_sample_size > 0:
            ids = np.random.choice(random_sample_size, num_poses, replace=True)
        else:
            raise ValueError("random_sample_size should be greater than or equal to 0.")
        # print(f"[INFO] Retrieve latent ids: {ids}")
        return self.Z[ids].reshape(num_poses, self.n)