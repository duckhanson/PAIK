# Import required packages
from datetime import datetime
from typing import Any
import numpy as np
import torch

from jrl.robot import Robot
from jrl.evaluation import solution_pose_errors

from utils.model import get_flow_model, get_knn
from utils.utils import load_all_data, data_preprocess_for_inference

from zuko.distributions import BoxUniform, DiagNormal
from zuko.flows import CNF, NSF, FlowModule, Unconditional

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
        'subnet_width': 1500,
        'subnet_num_layers': 3,
        'num_transforms': 16,
        'lr': 1.3e-4,
        'lr_weight_decay': 3.1e-2,
        'decay_step_size': 6e4,
        'gamma': 9e-2,
        'shrink_ratio': 0.61,
        'batch_size': 128,
        'num_epochs': 15,
        'ckpt_name': '1016-1439',
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
        
    def sample(self, C, K):
        return self._solver(C).sample((K,))
    
    def solve(self, single_pose: np.array, num_sols: int, return_numpy: bool=False):
        # Data Preprocessing
        C = data_preprocess_for_inference(P=single_pose, F=self._F, knn=self._knn)

        # Begin inference
        with torch.inference_mode():
            J_hat = self._solver(C).sample((num_sols,))
            J_hat = torch.reshape(J_hat, (num_sols, -1))
            
        if return_numpy:
            J_hat = J_hat.detach().cpu().numpy()
            
        return J_hat
    
    def _random_sample_poses(self, num_poses: int):
        # Randomly sample poses from test set
        idx = np.random.choice(self._P_ts.shape[0], num_poses, replace=False)
        P = self._P_ts[idx]
        return P
    
    def random_evaluation(self, num_poses: int, num_sols: int):
        # Randomly sample poses from test set
        P = self._random_sample_poses(num_poses=num_poses)
        
        # Data Preprocessing
        C = data_preprocess_for_inference(P=P, F=self._F, knn=self._knn)

        # Begin inference
        with torch.inference_mode():
            J_hat = self._solver(C).sample((num_sols,))
            J_hat = torch.reshape(J_hat, (num_poses, num_sols, -1))
            
        l2_errs = np.empty((J_hat.shape[0], J_hat.shape[1]))
        ang_errs = np.empty((J_hat.shape[0], J_hat.shape[1]))
        if P.shape[-1] == 3:
            P = np.column_stack((P, np.ones(shape=(len(P), 4))))
        for i, J in enumerate(J_hat):
            l2_errs[i], ang_errs[i] = solution_pose_errors(robot=self._robot, solutions=J, target_poses=P)
            
        l2_errs = l2_errs.flatten()
        ang_errs = ang_errs.flatten()

        # df = pd.DataFrame()
        # df['l2_errs'] = l2_errs
        # df['ang_errs'] = ang_errs
        # print(df.describe())
        return l2_errs.mean(), ang_errs.mean()
    
    def _update_solver(self):
        self._solver = FlowModule(
            transforms=self._solver.transforms,
            base=Unconditional(
                DiagNormal,
                torch.zeros((self._robot.n_dofs,)) + self._init_latent,
                torch.ones((self._robot.n_dofs,)) * self._shink_ratio,
                buffer=True,
            ),
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
        
    