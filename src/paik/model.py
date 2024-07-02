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


def get_flow_model(config: SolverConfig) -> tuple[Flow, Optimizer, ReduceLROnPlateau]:
    """
    Return flow model, optimizer, and scheduler

    Args:
        config (SolverConfig): defined in settings.py

    Returns:
        tuple[Flow, Optimizer, ReduceLROnPlateau]: flow model, optimizer, and scheduler
    """
    print(f"[INFO] create new model with config: {config}")
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
    # path_solver = f"{config.weight_dir}/{config.ckpt_name}.pth"
    # if config.enable_load_model and os.path.exists(path=path_solver):
    #     try:
    #         state = torch.load(path_solver)
    #         flow.load_state_dict(state["solver"])
    #         print(f"[SUCCESS] model load (solver) from {path_solver}.")
    #         optimizer.load_state_dict(state["opt"])
    #         print(f"[SUCCESS] model load (opt) from {path_solver}.")
    #     except Exception as e:
    #         print(f"[WARNING]{e}")
    #         print("[WARNING] try again will solve it :)")
    #         torch.save(
    #             {"solver": flow.state_dict(), "opt": optimizer.state_dict()},
    #             path_solver,
    #         )
    # else:
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

SUPPORTED_ROBOTS = [Panda, Fetch, FetchArm, Iiwa7]

def get_robot(robot_name: str, robot_dirs: Tuple[str, str, str, str]):
    """
    Return robot model, and create robot directories if not exists

    Args:
        robot_name (str): current only support "panda"
        robot_dirs (Tuple[str, str, str]): (data_dir, weight_dir, log_dir) defined in settings.py
    """
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
    else:
        raise NotImplementedError()
