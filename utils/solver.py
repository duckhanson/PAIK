# Import required packages
from datetime import datetime
from typing import Any
import numpy as np
import torch

from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm

from utils.model import get_flow_model, get_knn
from utils.utils import load_all_data, data_preprocess_for_inference

DEFAULT_SOLVER_PARAM_M3 = {
        'subnet_width': 1400,
        'subnet_num_layers': 3,
        'num_transforms': 9,
        'lr': 2.1e-4,
        'lr_weight_decay': 2.7e-2,
        'decay_step_size': 4e4,
        'gamma': 5e-2,
        'batch_size': 128,
        'num_epochs': 10,
        'ckpt_name': 'nsf',
    }

DEFAULT_SOLVER_PARAM_M7 = {
        'subnet_width': 1600,
        'subnet_num_layers': 3,
        'num_transforms': 16,
        'lr': 2.1e-4,
        'lr_weight_decay': 2.7e-2,
        'decay_step_size': 4e4,
        'gamma': 5e-2,
        'batch_size': 128,
        'num_epochs': 10,
        'ckpt_name': 'nsf',
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
                lr=solver_param["lr"],
                lr_weight_decay=solver_param["lr_weight_decay"],
                decay_step_size=solver_param["decay_step_size"],
                gamma=solver_param["gamma"],
                device='cuda',
                ckpt_name=solver_param["ckpt_name"])
        self._solver = flow
        self._optimizer = optimizer
        self._scheduler = scheduler
        
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
    
    @property
    def robot(self):
        return self._robot
    
    @property
    def param(self):
        return self._solver_param