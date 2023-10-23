# Import required packages
# import time
from __future__ import annotations

from os import path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.optim.lr_scheduler import StepLR
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional, MaskedAutoregressiveTransform
from zuko.flows.coupling import GeneralCouplingTransform
from zuko.flows.neural import UNAF
from zuko.flows.spline import NSF

from utils.settings import config
from utils.utils import save_pickle, load_pickle

def get_flow_model(
    enable_load_model: bool=True,
    num_transforms: int=config.num_transforms,
    subnet_width: int=config.subnet_width,
    subnet_num_layers: int=config.subnet_num_layers,
    shrink_ratio: float=config.shrink_ratio,
    lr: float=config.lr,
    lr_weight_decay: float=config.lr_weight_decay,
    decay_step_size: int=config.decay_step_size,
    gamma: float=config.decay_gamma,
    model_architecture: str='nsf',    
    optimizer_type: str= 'adamw',
    device=config.device,
    ckpt_name: str='',
    random_perm: bool=True,
    n: int=config.n,
    m: int=config.m,
    r: int=config.r,
):
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW, StepLR)
    :rtype: tuple
    """
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context
    num_conditions = m + r + 1
    if model_architecture == "nsf":
        flow = NSF(
            features=n,
            context=num_conditions,
            transforms=num_transforms,
            randperm=random_perm,
            bins=10,
            activation=config.activation,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(device)
    elif model_architecture == "nf":
        flow = Flow(
            transforms=[
                GeneralCouplingTransform(
                    features=n, 
                    context=num_conditions, 
                    activation=config.activation,
                    hidden_features=[subnet_width] * subnet_num_layers) for _ in range(num_transforms)
                # Unconditional(RotationTransform, torch.randn(n, n)), # type: ignore
            ],
            base=Unconditional(
                DiagNormal,
                torch.zeros(n),
                torch.ones(n),
                buffer=True,
            ), # type: ignore
        ).to(device)
    elif model_architecture == "unaf":
        flow = UNAF(
            features=n,
            context=num_conditions,
            transforms=num_transforms,
            randperm=random_perm,
            activation=config.activation,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(device)
    # elif config.architecture == "cnf":
    #     flow = CNF(
    #         features=config.n,
    #         context=config.num_conditions,
    #         transforms=num_transforms,
    #         activation=config.activation,
    #         hidden_features=[subnet_width] * subnet_num_layers,
    #     ).to(device)
    else:
        raise NotImplementedError("Not support architecture.")

    flow = get_sflow_model(flow, n=n, shrink_ratio=shrink_ratio, device=device)
    optimizer = get_optimizer(flow.parameters(), optimizer_type, lr, weight_decay=lr_weight_decay)
    path_solver = config.weight_dir + f"{ckpt_name}.pth"
    if enable_load_model and path.exists(path=path_solver):
        try:
            state = torch.load(path_solver)
            flow.load_state_dict(state_dict=state['solver'])
            optimizer.load_state_dict(state_dict=state['opt'])
            
            print(f"Model load successfully from {path_solver}")
        except Exception:
            print(f"Load err from {path_solver}, assuming you use different architecture.")
    else:
        print("Create a new model and start training.")

    # Train to maximize the log-likelihood
    scheduler = StepLR(optimizer, step_size=decay_step_size, gamma=gamma)

    return flow, optimizer, scheduler

def get_optimizer(params, optimizer_type, lr, weight_decay=0.0):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "sgd_nesterov":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    raise NotImplementedError

def get_sflow_model(flow: NSF | Flow, n: int, shrink_ratio: float = config.shrink_ratio, device: str = config.device):
    """
    shrink normal distribution model

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
    sflow = Flow(
        transforms=flow.transforms, # type: ignore
        base=Unconditional(
            DiagNormal,
            torch.zeros((n,)),
            torch.ones((n,)) * shrink_ratio,
            buffer=True,
        ), # type: ignore
    )

    sflow.to(device)

    return sflow


def get_knn(P_tr: np.ndarray, n: int=config.n, m: int=config.m, r: int=config.r):
    """
    fit a knn model

    Parameters
    ----------
    P_tr : np.ndarray
        end-effector positions of training data

    Returns
    -------
    NearestNeighbors
        a knn model that fit end-effector positions of training data
    """
    path = config.path_knn(n, m, r)
    try:
        knn = load_pickle(file_path=path)
        print(f"knn load successfully from {path}")
    except FileNotFoundError as e:
        print(e)
        knn = NearestNeighbors(n_neighbors=1)  
        P_tr = np.atleast_2d(P_tr)
        knn.fit(P_tr[:, :3])  
        save_pickle(file_path=path, obj=knn)
        print(f"Create and save knn at {path}.")
    
    return knn