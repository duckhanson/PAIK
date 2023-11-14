# Import required packages
from __future__ import annotations
import os
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from klampt.model import trajectory
from jrl.robot import Robot
# from jrl.evaluation import solution_pose_errors, evaluate_solutions

from utils.settings import config as cfg
from utils.model import get_flow_model, get_knn, get_robot
from utils.utils import load_numpy, save_numpy
from utils.dataset import load_all_data, data_preprocess_for_inference, nearest_neighbor_F
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional

DEFAULT_SOLVER_PARAM_M3 = {
        'subnet_width': 1024,
        'subnet_num_layers': 3,
        'num_transforms': 10,
        'lr': 3.3e-4,
        'lr_weight_decay': 1.3e-2,
        'decay_step_size': 7e4,
        'gamma': 9.4e-2,
        'shrink_ratio': 0.61,
        'batch_size': 128,
        'num_epochs': 10,
        'model_architecture': 'nsf',
        'noise_esp': 1e-3,
        'noise_esp_decay': 0.8,
        'opt_type': 'adamw',
        'sche_type': 'steplr',
        'ckpt_name': '0930-0346',
        'nmr': (7, 3, 4),
        'random_perm': True,
        'enable_load_model': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

DEFAULT_SOLVER_PARAM_M7 = {
        'lr': 3.6e-4,
        'gamma': 8.4e-2,
        'opt_type': 'adamw',
        'noise_esp': 1.9e-3,
        'sche_type': 'plateau',
        'batch_size': 128,
        'num_epochs': 15,
        'random_perm': True,
        'subnet_width': 1150,
        'num_transforms': 8,
        'decay_step_size': 5e4,
        'lr_weight_decay': 1.8e-2,
        'noise_esp_decay': 0.92,
        'subnet_num_layers': 3,
        'model_architecture': 'nsf',
        'shrink_ratio': 0.61,
        'ckpt_name': '1107-1013',
        'nmr': (7, 7, 1),
        'enable_load_model': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

class Solver:
    def __init__(self, robot: Robot = get_robot(), solver_param: dict = DEFAULT_SOLVER_PARAM_M7) -> None:
        self._robot = robot
        self._solver_param = solver_param
        self._device = solver_param['device']
        # Neural spline flow (NSF) with 3 sample features and 5 context features
        n, m, r = solver_param['nmr']
        flow, optimizer, scheduler = get_flow_model(
                enable_load_model=True,
                num_transforms=solver_param["num_transforms"],
                subnet_width=solver_param["subnet_width"],
                subnet_num_layers=solver_param["subnet_num_layers"],
                shrink_ratio=solver_param["shrink_ratio"],
                lr=solver_param["lr"],
                lr_weight_decay=solver_param["lr_weight_decay"],
                decay_step_size=solver_param["decay_step_size"],
                gamma=solver_param["gamma"],
                optimizer_type=solver_param['opt_type'],
                scheduler_type=solver_param['sche_type'],
                device=self._device,
                model_architecture=solver_param["model_architecture"],
                ckpt_name=solver_param["ckpt_name"],
                random_perm=solver_param['random_perm'],
                n=n,
                m=m,
                r=r) # type: ignore
        self._solver = flow
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._shink_ratio = solver_param["shrink_ratio"]
        self._init_latent = torch.zeros((1, self.robot.n_dofs)).to(self._device)
        
        # load inference data
        assert n == self._robot.n_dofs, f"n should be {self._robot.n_dofs} as the robot"
        self._J_tr, self._P_tr, self._P_ts, self._F = load_all_data(self._robot, n=n, m=m, r=r)
        self._knn = get_knn(P_tr=self._P_tr, n=n, m=m, r=r)
        self._m = m
    
    @property
    def latent(self):
        return self._init_latent
    
    @property
    def shrink_ratio(self):
        return self._shink_ratio
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def param(self):
        return self._solver_param
    
    @shrink_ratio.setter
    def shrink_ratio(self, value: float):
        assert value >= 0 and value < 1
        self._shink_ratio = value
        self.__update_solver()
        
    @latent.setter
    def latent(self, value: torch.Tensor):
        assert value.shape == (1, self._robot.n_dofs)
        self._init_latent = value
        self.__update_solver()
        
    
    def sample(self, C, K):
        with torch.inference_mode():
            return self._solver(C).sample((K,)).detach().cpu()
        
    
    def sample_n(self, C, K):
        num_poses = C.shape[0]
        num_sols = K
        batch_size = 100
        C = C.view(1, *C.shape).tile(K, 1, 1)
        C = C.reshape(-1, batch_size, C.shape[-1])
        S = torch.empty((num_sols * num_poses, self._robot.n_dofs), device=self._device)
        S = S.reshape(-1, batch_size, S.shape[-1])
        for i, c in enumerate(C):
            with torch.inference_mode():
                S[i] = self._solver(c).sample()
        return S.reshape(num_sols, num_poses, -1)
    
    def solve(self, single_pose: np.ndarray, num_sols: int, k: int=1, return_numpy: bool=False):
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
        C = data_preprocess_for_inference(P=single_pose, F=self._F, knn=self._knn, m=self._m, k=k)

        # Begin inference
        J_hat = self.sample(C, num_sols)
                
        J_hat = torch.reshape(J_hat, (num_sols * k, -1))
            
        if return_numpy:
            J_hat = J_hat.detach().cpu().numpy()
            
        return J_hat
    
    def __random_sample_poses(self, num_poses: int):
        # Randomly sample poses from test set
        idx = np.random.choice(self._P_ts.shape[0], num_poses, replace=False)
        P = self._P_ts[idx]
        return P
    
    def random_evaluation(self, num_poses: int, num_sols: int, return_time: bool=False):
        # Randomly sample poses from test set
        P = self.__random_sample_poses(num_poses=num_poses)
        time_begin = time()
        # Data Preprocessing
        C = data_preprocess_for_inference(P=P, F=self._F, knn=self._knn, m=self._m)

        # Begin inference
        J_hat = self.sample(C, num_sols)
        J_hat = J_hat.detach().cpu().numpy()
        inference_time = round((time() - time_begin), 3)    
        print(f"model inference time: {inference_time}")
        
        P = P if self._m == 7 else np.column_stack((P, np.ones(shape=(len(P), 4))))
         
        l2_errs = np.empty((num_poses, num_sols))
        ang_errs = np.empty((num_poses, num_sols))
        for i in range(num_poses):
            l2_errs[i], ang_errs[i] = solution_pose_errors(robot=self._robot, solutions=J_hat[:, i, :], target_poses=P[i], device=self._device)
            
        errors_time = round((time() - time_begin), 3) - inference_time
        print(f"calculation errors time: {errors_time}")
        
        avg_l2_errs = l2_errs.mean()
        avg_ang_errs = ang_errs.mean()
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        if return_time:
            return avg_l2_errs, avg_ang_errs, avg_inference_time
        else:
            return avg_l2_errs, avg_ang_errs
    
    def __update_solver(self):
        self._solver = Flow(
            transforms=self._solver.transforms, # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,), device=self._device) + self._init_latent,
                torch.ones((self._robot.n_dofs,), device=self._device) * self._shink_ratio,
                buffer=True,
            ), # type: ignore
        )
        
    def sample_P_path(self, load_time: str = "", num_steps=20, seed=47) -> np.ndarray:
        """
        sample a path from P_ts

        Parameters
        ----------
        load_time : str, optional
            file name of load P_path, by default ""
        num_steps : int, optional
            length of the generated path, by default 20

        Returns
        -------
        np.ndarray
            array_like(num_steps, m)
        """
        np.random.seed(seed)
        
        if load_time == "":
            traj_dir = cfg.traj_dir + datetime.now().strftime("%m%d%H%M%S") + "/"
        else:
            traj_dir = cfg.traj_dir + load_time + "/"

        P_path_file_path = traj_dir + "ee_traj.npy"

        if load_time == "" or not os.path.exists(path=P_path_file_path):
            # endPoints = np.random.rand(2, cfg.m) # 2 for begin and end
            rand_idxs = np.random.randint(low=0, high=len(self._P_ts), size=2)
            endPoints = self._P_ts[rand_idxs]
            traj = trajectory.Trajectory(milestones=endPoints) # type: ignore
            P_path = np.empty((num_steps, self._m))
            for i in range(num_steps):
                iStep = i/num_steps
                point = traj.eval(iStep)
                P_path[i] = point
            
            save_numpy(file_path=P_path_file_path, arr=P_path)
        else:
            P_path = load_numpy(file_path=P_path_file_path)

        if os.path.exists(path=traj_dir):
            print(f"{traj_dir} load successfully.")
        
        return P_path
    
    def __sample_J_traj(self, P_path: np.ndarray, ref_F: np.ndarray):
        """
        sample a trajectory from IK solver that fit P_path

        Parameters
        ----------
        P_path : np.ndarray
            a sequence of target end-effector poses
        ref_F : np.ndarray
            posture features

        Returns
        -------
        torch.Tensor
            array_like(num_steps, n_dofs)
        """
        assert self._shink_ratio < 0.2, "shrink_ratio should be less than 0.2"
        
        P_path = P_path[:, :self._m]
        ref_F = np.atleast_2d(ref_F)
        if len(P_path) != len(ref_F):        
            ref_F = np.tile(ref_F, (len(P_path), 1)) # type: ignore

        C = np.column_stack((P_path, ref_F, np.zeros((len(P_path),)))) # type: ignore
        C = torch.from_numpy(C).float().to(self._device) # type: ignore

        J_hat = self.sample(C, 1)
        J_hat = torch.reshape(J_hat, (-1, self._robot.n_dofs))
        return J_hat
    
    def __sample_n_J_traj(self, P_path: np.ndarray, ref_F: np.ndarray):
        """
        sample a trajectory from IK solver that fit P_path

        Parameters
        ----------
        P_path : np.ndarray
            a sequence of target end-effector poses
        ref_F : np.ndarray
            posture features

        Returns
        -------
        torch.Tensor
            array_like(num_steps, n_dofs)
        """
        assert self._shink_ratio < 0.2, "shrink_ratio should be less than 0.2"
        num_traj = len(ref_F)
        num_steps = len(P_path)
        
        if self._m == 3:
            P_path = P_path[:, :self._m]
        
        C = np.empty((num_traj, num_steps, self._m + ref_F.shape[-1] + 1))
        for i, f in enumerate(ref_F):
            f = np.tile(np.atleast_2d(f), (num_steps, 1)) # type: ignore
            C[i] = np.column_stack((P_path, f, np.zeros((num_steps,)))) # type: ignore
        C = C.reshape(-1, self._m + ref_F.shape[-1] + 1)
        C = torch.from_numpy(C).float().to(self._device) # type: ignore

        J_hat = self.sample(C, 1)
        J_hat = torch.reshape(J_hat, (num_traj, num_steps, self._robot.n_dofs))
        return J_hat
    
    def __max_joint_angle_change(self, qs: torch.Tensor | np.ndarray):
        if isinstance(qs, torch.Tensor):
            qs = qs.detach().cpu().numpy()
        return np.rad2deg(np.max(np.abs(np.diff(qs, axis=0))))
    
    def path_following(
        self,
        load_time: str = "", 
        num_traj: int = 3,
        num_steps=20,
        shrink_ratio: float = 0.1,
        enable_evaluation: bool = False,
        enable_plot: bool = False,
        seed: int = 47
    ):
        """
        evaluate the performance of path following

        Parameters
        ----------
        load_time : str, optional
            file name of load P_path, by default ""
        num_traj : int, optional
            number of demo trajectories, by default 3
        num_steps : int, optional
            length of the generated path, by default 20
        shrink_ratio : float, optional
            the shrink ratio of the based distribution of IK solver, by default 0.1
        enable_evaluation : bool, optional
            use evaluation or not, by default False
        enable_plot : bool, optional
            use plot or not, by default False
        """
        old_shink_ratio = self._shink_ratio
        self.shrink_ratio = shrink_ratio
        
        # print(f'using shrink_ratio: {self.shrink_ratio}')
        
        P_path = self.sample_P_path(load_time=load_time, num_steps=num_steps, seed=seed)
        P_path_7 = P_path if self._m == 7 else np.column_stack((P_path, np.ones((len(P_path), 4)))) # type: ignore
        
        # ref_F = nearest_neighbor_F(self._knn, np.atleast_2d(P_path[0]), self._F, n_neighbors=300) # type: ignore # knn
        # nn1_F = nearest_neighbor_F(self._knn, np.atleast_2d(P_path), self._F, n_neighbors=1) # type: ignore # knn
        time_begin = time()
        
        ref_F = nearest_neighbor_F(self._knn, np.atleast_2d(P_path), self._F, n_neighbors=30) # type: ignore # knn
        ref_F = ref_F.flatten()
        rand_idxs = np.random.randint(low=0, high=len(ref_F), size=num_traj)
        ref_F = ref_F[rand_idxs].reshape(num_traj, -1)
        # ref_F = F
        # ref_F = rand_F(Path[0], F)
        
        # rand_idxs = list(range(num_traj))
        # qs = self.__sample_J_traj(P_path, nn1_F)
        # print("="*6 + f"=(use nearest)" + "="*6)
        # print(P_path_7.shape)
        # l2_err, ang_err = solution_pose_errors(robot=self._robot, solutions=qs, target_poses=P_path_7)
        # df = pd.DataFrame({'l2_err': l2_err, 'ang_err': ang_err})
        # print(df.describe())
        l2_err_arr = np.empty((num_traj, num_steps))
        ang_err_arr = np.empty((num_traj, num_steps))
        
        Qs = self.__sample_n_J_traj(P_path, ref_F)
        Qs = Qs.detach().cpu().numpy()
        if enable_evaluation:
            mjac_arr = np.array([self.__max_joint_angle_change(qs) for qs in Qs])
            for i in range(num_traj):
                l2_err_arr[i], ang_err_arr[i] = solution_pose_errors(robot=self._robot, solutions=Qs[i], target_poses=P_path_7, device=self._device)
            df = pd.DataFrame({'l2_err': l2_err_arr.mean(axis=1), 'ang_err': ang_err_arr.mean(axis=1), 'mjac': mjac_arr})
            print(df.describe())
            print(f"avg_inference_time: {round((time() - time_begin) / num_traj, 3)}")
            
            
        if enable_plot:
            return P_path_7, Qs, ref_F

        self.shrink_ratio = old_shink_ratio

from jrl.evaluation import _get_target_pose_batch
from jrl.conversions import geodesic_distance_between_quaternions

def solution_pose_errors(
    robot: Robot, solutions: np.ndarray, target_poses: torch.Tensor | np.ndarray, device: str
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
    ang_errors = geodesic_distance_between_quaternions(q_target_pt, q_current_pt).detach().cpu().numpy()
    return l2_errors, ang_errors
    
    