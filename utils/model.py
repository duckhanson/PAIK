# Import required packages
# import time
from os import path

import numpy as np
# import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
# from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from zuko.distributions import BoxUniform, DiagNormal
from zuko.flows import CNF, NSF, FlowModule, Unconditional

from utils.settings import config
# from tqdm import tqdm, trange
# from utils.dataset import create_dataset
# from utils.robot import Robot
from utils.utils import save_pickle

# import zuko


def get_flow_model(
    load_model=True,
    num_transforms=config.num_transforms,
    subnet_width=config.subnet_width,
    subnet_num_layers=config.subnet_num_layers,
    shrink_ratio=config.shrink_ratio,
    lr=config.lr,
    lr_weight_decay=config.lr_weight_decay,
    decay_step_size=config.decay_step_size,
    gamma=config.decay_gamma,
):
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW, StepLR)
    :rtype: tuple
    """
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context
    if config.architecture == "nsf":
        flow = NSF(
            features=config.n,
            context=config.num_conditions,
            transforms=num_transforms,
            randperm=True,
            activation=config.activation,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(config.device)
    elif config.architecture == "cnf":
        flow = CNF(
            features=config.n,
            context=config.num_conditions,
            transforms=num_transforms,
            activation=config.activation,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(config.device)
    else:
        raise NotImplementedError("Not support architecture.")

    flow = get_sflow_model(flow, shrink_ratio=shrink_ratio)

    if load_model and path.exists(path=config.path_solver):
        try:
            flow.load_state_dict(state_dict=torch.load(config.path_solver))
            print(f"Model load successfully from {config.path_solver}")
        except Exception:
            print("Load err, assuming you use different architecture.")
    else:
        print("Create a new model and start training.")

    # Train to maximize the log-likelihood
    optimizer = AdamW(flow.parameters(), lr=lr, weight_decay=lr_weight_decay)
    scheduler = StepLR(optimizer, step_size=decay_step_size, gamma=gamma)

    return flow, optimizer, scheduler


def get_iflow_model(
    flow: NSF, init_sample: torch.Tensor, shrink_ratio: float = 0.01
) -> FlowModule:
    """
    sampling from initial samples as search space centers
    seach shrink_ratio region

    :param flow: _description_
    :type flow: NSF
    :param init_sample: _description_
    :type init_sample: torch.Tensor
    :param shrink_ratio: _description_, defaults to 0.01
    :type shrink_ratio: float, optional
    :return: _description_
    :rtype: FlowModule
    """
    iflow = FlowModule(
        transforms=flow.transforms,
        base=Unconditional(
            DiagNormal,
            torch.zeros((config.n,)) + init_sample,
            torch.ones((config.n,)) * shrink_ratio,
            buffer=True,
        ),
    )

    iflow.to(config.device)

    return iflow


def get_sflow_model(flow: NSF, shrink_ratio: float = config.shrink_ratio):
    """
    shrink normal distribution model

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
    sflow = FlowModule(
        transforms=flow.transforms,
        base=Unconditional(
            DiagNormal,
            torch.zeros((config.n,)),
            torch.ones((config.n,)) * shrink_ratio,
            buffer=True,
        ),
    )

    sflow.to(config.device)

    return sflow


def get_nflow_model(flow: NSF, shrink_ratio=config.shrink_ratio):
    """
    _summary_

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
    nflow = FlowModule(
        transforms=flow.transforms,
        base=Unconditional(
            BoxUniform,
            -torch.ones((config.n,)) * shrink_ratio,
            torch.ones((config.n,)) * shrink_ratio,
            buffer=True,
        ),
    )

    nflow.to(config.device)

    return nflow

def get_knn(P_tr: np.array):
    """
    fit a knn model

    Parameters
    ----------
    P_tr : np.array
        end-effector positions of training data

    Returns
    -------
    NearestNeighbors
        a knn model that fit end-effector positions of training data
    """
    
    knn = NearestNeighbors(radius=0.07)  
    knn.fit(P_tr)  
    save_pickle(file_path=config.path_knn, obj=knn)
    
    return knn