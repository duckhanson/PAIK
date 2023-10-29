# Import required packages
# import time
from __future__ import annotations

from os import path
import numpy as np
import torch
from torch.nn import LeakyReLU
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional, MaskedAutoregressiveTransform
from zuko.flows.coupling import GeneralCouplingTransform
from zuko.flows.neural import UNAF
from zuko.flows.spline import NSF

from utils.settings import config
from utils.utils import save_pickle, load_pickle

DEFAULT_ACTIVATION = LeakyReLU

def get_flow_model(
    num_transforms: int,
    subnet_width: int,
    subnet_num_layers: int,
    shrink_ratio: float,
    lr: float,
    lr_weight_decay: float,
    decay_step_size: int,
    gamma: float,
    model_architecture: str,    
    optimizer_type: str,
    scheduler_type: str,
    device: str,
    ckpt_name: str,
    n: int,
    m: int,
    r: int,
    random_perm: bool,
    enable_load_model: bool,
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
            activation=DEFAULT_ACTIVATION,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(device)
    elif model_architecture == "nf":
        flow = Flow(
            transforms=[
                GeneralCouplingTransform(
                    features=n, 
                    context=num_conditions, 
                    activation=DEFAULT_ACTIVATION,
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
            activation=DEFAULT_ACTIVATION,
            hidden_features=[subnet_width] * subnet_num_layers,
        ).to(device)
    # elif architecture == "cnf":
    #     flow = CNF(
    #         features=n,
    #         context=num_conditions,
    #         transforms=num_transforms,
    #         activation=DEFAULT_ACTIVATION,
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
    if scheduler_type == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=lr * 1e-2)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2, verbose=True)

    return flow, optimizer, scheduler

def get_optimizer(params, optimizer_type, lr, weight_decay):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "sgd_nesterov":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    raise NotImplementedError

def get_sflow_model(flow: NSF | Flow, n: int, shrink_ratio: float, device: str):
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


def get_knn(P_tr: np.ndarray, n: int, m: int, r: int):
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