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
from paik.settings import SolverConfig
from jrl.robots import Panda


def get_flow_model(config: SolverConfig):
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW, StepLR)
    :rtype: tuple
    """
    assert config.model_architecture in ["nsf"]
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context

    flow = change_flow_base(
        NSF(
            features=config.n,
            # number of conditions
            context=config.m + 1 if config.use_nsf_only else config.m + config.r + 1,
            transforms=config.num_transforms,
            randperm=config.randperm,
            bins=config.num_bins,
            activation=LeakyReLU,
            hidden_features=[config.subnet_width] * config.subnet_num_layers,
        ),
        n=config.n,
        base_std=config.base_std,
    )
    flow = flow.to(config.device)

    optimizer = optim.AdamW(
        flow.parameters(),
        lr=config.lr,
        weight_decay=config.lr_weight_decay,
        amsgrad=config.lr_amsgrad,
        betas=config.lr_beta,
    )
    path_solver = f"{config.weight_dir}/{config.ckpt_name}.pth"
    if config.enable_load_model and os.path.exists(path=path_solver):
        try:
            state = torch.load(path_solver)
            flow.load_state_dict(state["solver"])
            print(f"[SUCCESS] model load (solver) from {path_solver}.")
            optimizer.load_state_dict(state["opt"])
            print(f"[SUCCESS] model load (opt) from {path_solver}.")
        except Exception as e:
            print(f"[WARNING]{e}")
            print("[WARNING] try again will solve it :)")
            torch.save(
                {"solver": flow.state_dict(), "opt": optimizer.state_dict()},
                path_solver,
            )
    else:
        print("[WARNING] create a new model and start training.")

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


def change_flow_base(flow: NSF | Flow, n: int, base_std: float):
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
            torch.ones((n,)) * base_std,
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
