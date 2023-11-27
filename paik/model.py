# Import required packages
# import time
from __future__ import annotations
from typing import Tuple

import os
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

from jrl.robots import Panda
from paik.utils import save_pickle, load_pickle

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
    path_solver: str,
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
                    hidden_features=[subnet_width] * subnet_num_layers,
                )
                for _ in range(num_transforms)
                # Unconditional(RotationTransform, torch.randn(n, n)), # type: ignore
            ],
            base=Unconditional(
                DiagNormal,
                torch.zeros(n),
                torch.ones(n),
                buffer=True,
            ),  # type: ignore
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
    optimizer = get_optimizer(
        flow.parameters(), optimizer_type, lr, weight_decay=lr_weight_decay
    )
    if enable_load_model and os.path.exists(path=path_solver):
        try:
            state = torch.load(path_solver)
            flow.load_state_dict(state_dict=state["solver"])
            optimizer.load_state_dict(state_dict=state["opt"])

            print(f"Model load successfully from {path_solver}")
        except Exception as e:
            print(f"[Warning] load error from {path_solver}, {e}.")
    else:
        print("Create a new model and start training.")

    # Train to maximize the log-likelihood
    if scheduler_type == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=3, eta_min=lr * 1e-2)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=2, verbose=True
        )
    else:
        scheduler = StepLR(optimizer, step_size=decay_step_size, gamma=gamma)

    return flow, optimizer, scheduler


def get_optimizer(params, optimizer_type, lr, weight_decay):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "sgd_nesterov":
        return optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True
        )

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
        transforms=flow.transforms,  # type: ignore
        base=Unconditional(
            DiagNormal,
            torch.zeros((n,)),
            torch.ones((n,)) * shrink_ratio,
            buffer=True,
        ),  # type: ignore
    )

    sflow.to(device)

    return sflow


def get_knn(P_tr: np.ndarray, path: str):
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
    try:
        knn = load_pickle(file_path=path)
        print(f"knn load successfully from {path}")
    except FileNotFoundError as e:
        print(e)
        knn = NearestNeighbors(n_neighbors=1)
        P_tr = np.atleast_2d(P_tr)
        # knn.fit(P_tr[:, :3])
        knn.fit(P_tr)
        save_pickle(file_path=path, obj=knn)
        print(f"Create and save knn at {path}.")

    return knn


def get_robot(robot_name: str, robot_dirs: Tuple[str, str, str]):
    # def create_robot_dirs(dir_paths) -> None:
    for dp in robot_dirs:
        if not os.path.exists(path=dp):
            os.makedirs(name=dp)
            print(f"Create {dp}")

    if robot_name == "panda":
        return Panda()
    else:
        raise NotImplementedError()
