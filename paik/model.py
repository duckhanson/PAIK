# Import required packages
from __future__ import annotations
from typing import Tuple

import os
import numpy as np
import torch
from torch.nn import LeakyReLU
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional
from zuko.flows.spline import NSF

from jrl.robots import Panda

DEFAULT_ACTIVATION = LeakyReLU


def get_flow_model(
    num_transforms: int,
    subnet_width: int,
    subnet_num_layers: int,
    shrink_ratio: float,
    lr: float,
    lr_weight_decay: float,
    gamma: float,
    model_architecture: str,
    device: str,
    path_solver: str,
    n: int,
    m: int,
    r: int,
    random_perm: bool,
    enable_load_model: bool,
    disable_posture_feature: bool,
    shce_patience: int,
):
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW, StepLR)
    :rtype: tuple
    """
    assert model_architecture in ["nsf"]
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context
    num_conditions = m + 1 if disable_posture_feature else m + r + 1

    flow = change_flow_base(
        NSF(
            features=n,
            context=num_conditions,
            transforms=num_transforms,
            randperm=random_perm,
            bins=10,
            activation=DEFAULT_ACTIVATION,
            hidden_features=[subnet_width] * subnet_num_layers,
        ),
        n=n,
        shrink_ratio=shrink_ratio,
    )
    flow = flow.to(device)

    optimizer = optim.AdamW(flow.parameters(), lr=lr, weight_decay=lr_weight_decay)
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
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=gamma, patience=shce_patience, verbose=True
    )

    return flow, optimizer, scheduler


def change_flow_base(flow: NSF | Flow, n: int, shrink_ratio: float):
    """
    shrink normal distribution model

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
    return Flow(
        transforms=flow.transforms,  # type: ignore
        base=Unconditional(
            DiagNormal,
            torch.zeros((n,)),
            torch.ones((n,)) * shrink_ratio,
            buffer=True,
        ),  # type: ignore
    )


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
