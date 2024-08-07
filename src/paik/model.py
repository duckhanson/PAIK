# Import required packages
from __future__ import annotations
from typing import Tuple

import os
import numpy as np
import torch
from torch.nn import LeakyReLU
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zuko.distributions import DiagNormal
from zuko.flows import Flow, Unconditional
from zuko.flows.spline import NSF
from .settings import SolverConfig
from jrl.robots import Panda, Fetch, FetchArm, Iiwa7
from klampt_robot import AtlasArm, AtlasWaistArm, BaxterArm, PR2
from pprint import pprint


def get_flow_model(config: SolverConfig) -> tuple[Flow, Optimizer, ReduceLROnPlateau]:
    """
    Return flow model, optimizer, and scheduler

    Args:
        config (SolverConfig): defined in settings.py

    Returns:
        tuple[Flow, Optimizer, ReduceLROnPlateau]: flow model, optimizer, and scheduler
    """
    pprint(f"[INFO] create new model with config: {config}")
    assert config.model_architecture in ["nsf"]
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context

    flow = Flow(
        transforms=NSF(
            features=config.n,
            # number of conditions
            context=config.m + 1 if config.use_nsf_only else config.m + config.r + 1,
            transforms=config.num_transforms,
            randperm=config.randperm,
            bins=config.num_bins,
            activation=LeakyReLU,
            hidden_features=[config.subnet_width] * config.subnet_num_layers,
        ).transforms,  # type: ignore
        base=Unconditional(
            DiagNormal,
            torch.zeros((config.n,)),
            torch.ones((config.n,)) * config.base_std,
            buffer=True,
        ),  # type: ignore
    ).to(config.device)

    optimizer = AdamW(
        flow.parameters(),
        lr=config.lr,
        weight_decay=config.lr_weight_decay,
        amsgrad=config.lr_amsgrad,
        betas=config.lr_beta,
    )

    print("[WARNING] not load model yet.")

    # Train to maximize the log-likelihood
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.gamma,
        patience=config.shce_patience,
        eps=1e-10,
        verbose=True,
    )

    return flow, optimizer, scheduler

SUPPORTED_ROBOTS = ["panda", "fetch", "fetch_arm", "iiwa7", "atlas_arm", "atlas_waist_arm", "baxter_arm", "pr2"]

def get_robot(robot_name: str, robot_dirs: Tuple[str, str, str, str]):
    """
    Return robot model, and create robot directories if not exists

    Args:
        robot_name (str): current only support "panda"
        robot_dirs (Tuple[str, str, str]): (data_dir, weight_dir, log_dir) defined in settings.py
    """
    if robot_name not in SUPPORTED_ROBOTS:
        raise NotImplementedError(f"Robot {robot_name} is not supported yet.")
    
    # def create_robot_dirs(dir_paths) -> None:
    for dp in robot_dirs:
        if not os.path.exists(path=dp):
            os.makedirs(name=dp)
            print(f"Create {dp}")

    if robot_name == "panda":
        return Panda()
    elif robot_name == "fetch":
        return Fetch()
    elif robot_name == "fetch_arm":
        return FetchArm()
    elif robot_name == "iiwa7":
        return Iiwa7()
    elif robot_name == "atlas_arm":
        return AtlasArm()
    elif robot_name == "atlas_waist_arm":
        return AtlasWaistArm()
    elif robot_name == "baxter_arm":
        return BaxterArm()
    elif robot_name == "pr2":
        return PR2()
    else:
        raise NotImplementedError()
