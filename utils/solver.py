# Import required packages
from time import time
from typing import Any
import numpy as np
import torch

from jrl.robot import Robot
from jrl.evaluation import solution_pose_errors

from utils.model import get_flow_model, get_knn
from utils.utils import load_all_data, data_preprocess_for_inference

from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional

DEFAULT_SOLVER_PARAM_M3 = {
        'subnet_width': 1400,
        'subnet_num_layers': 3,
        'num_transforms': 9,
        'lr': 2.1e-4,
        'lr_weight_decay': 2.7e-2,
        'decay_step_size': 4e4,
        'gamma': 5e-2,
        'shrink_ratio': 0.61,
        'batch_size': 128,
        'num_epochs': 10,
        'ckpt_name': 'nsf',
    }

DEFAULT_SOLVER_PARAM_M7 = {
        'subnet_width': 1024,
        'subnet_num_layers': 3,
        'num_transforms': 14,
        'lr': 1.3e-4,
        'lr_weight_decay': 3.1e-2,
        'decay_step_size': 6e4,
        'gamma': 9e-2,
        'shrink_ratio': 0.61,
        'batch_size': 128,
        'num_epochs': 15,
        'ckpt_name': '1018-0133',
    }

class Solver:
    def __init__(self, robot: Robot, solver_param: dict = DEFAULT_SOLVER_PARAM_M3) -> None:
        self._robot = robot
        self._solver_param = solver_param
        # Neural spline flow (NSF) with 3 sample features and 5 context features
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
                device='cuda',
                ckpt_name=solver_param["ckpt_name"])
        self._solver = flow
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._shink_ratio = solver_param["shrink_ratio"]
        self._init_latent = torch.zeros((1, self.robot.n_dofs)).cuda()
        
        # load inference data
        self._J_tr, self._P_tr, self._P_ts, self._F = load_all_data(self._robot)
        self._knn = get_knn(P_tr=self._P_tr)
        self.num_conditions = self._P_tr.shape[-1]
        
    def sample(self, C, K):
        with torch.inference_mode():
            return self._solver(C).sample((K,))
    
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
        C = data_preprocess_for_inference(P=single_pose, F=self._F, knn=self._knn, k=k)

        # Begin inference
        J_hat = self.sample(C, num_sols)
                
        J_hat = torch.reshape(J_hat, (num_sols * k, -1))
            
        if return_numpy:
            J_hat = J_hat.detach().cpu().numpy()
            
        return J_hat
    
    def _random_sample_poses(self, num_poses: int):
        # Randomly sample poses from test set
        idx = np.random.choice(self._P_ts.shape[0], num_poses, replace=False)
        P = self._P_ts[idx]
        return P
    
    def random_evaluation(self, num_poses: int, num_sols: int, return_time: bool=False):
        # Randomly sample poses from test set
        P = self._random_sample_poses(num_poses=num_poses)
        time_begin = time()
        # Data Preprocessing
        C = data_preprocess_for_inference(P=P, F=self._F, knn=self._knn)

        # Begin inference
        J_hat = self.sample(C, num_sols)
            
        l2_errs = np.empty((num_poses, num_sols))
        ang_errs = np.empty((num_poses, num_sols))
        if self.num_conditions == 3:
            P = np.column_stack((P, np.ones(shape=(len(P), 4))))
        for i in range(num_poses):

            l2_errs[i], ang_errs[i] = solution_pose_errors(robot=self._robot, solutions=J_hat[:, i, :], target_poses=P[i])
            
        l2_errs = l2_errs.flatten()
        ang_errs = ang_errs.flatten()
        
        avg_inference_time = round((time() - time_begin) / num_poses, 3)

        # # df = pd.DataFrame()
        # # df['l2_errs'] = l2_errs
        # # df['ang_errs'] = ang_errs
        # # print(df.describe())
        if return_time:
            return l2_errs.mean(), ang_errs.mean(), avg_inference_time
        else:
            return l2_errs.mean(), ang_errs.mean()
    
    def _update_solver(self):
        self._solver = Flow(
            transforms=self._solver.transforms, # type: ignore
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,)) + self._init_latent,
                torch.ones((self._robot.n_dofs,)) * self._shink_ratio,
                buffer=True,
            ), # type: ignore
        )
    
    @property
    def latent(self):
        return self._init_latent
    
    @property
    def shirnk_ratio(self):
        return self._shink_ratio
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def param(self):
        return self._solver_param
    
    @shirnk_ratio.setter
    def shirnk_ratio(self, value: float):
        assert value >= 0 and value < 1
        self._shink_ratio = value
        self._update_solver()
        
    @latent.setter
    def latent(self, value: torch.Tensor):
        assert value.shape == (1, self._robot.n_dofs)
        self._init_latent = value
        self._update_solver()
        
    