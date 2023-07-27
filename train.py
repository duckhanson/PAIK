# Import required packages
# import time
# from os import path

import flaml
# import numpy as np
# import pandas as pd
import torch
from ray import tune
# import zuko
# from hnne import HNNE
# from torch import Tensor, nn
from tqdm import tqdm

import wandb
# from utils.dataset import create_dataset
from utils.model import *
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import *

# from zuko.distributions import BoxUniform, DiagNormal, Minimum
# from zuko.flows import NSF, Distribution, DistributionModule, FlowModule, Unconditional


def train_step(model, batch, optimizer, scheduler):
    """
    _summary_

    Args:
        model (_type_): _description_
        batch (_type_): _description_
        optimizer (_type_): _description_
        scheduler (_type_): _description_

    Returns:
        _type_: _description_
    """
    x, y = add_noise(batch)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def mini_train(robot, num_epochs, sweep_config):
    # data generation
    J_tr, P_tr = data_collection(robot=robot, N=cfg.N_train)
    J_ts, P_ts = data_collection(robot=robot, N=cfg.N_test)
    F = posture_feature_extraction(J_tr)
    train_loader = get_train_loader(J=J_tr, P=P_tr, F=F)
    # Build Generative model, NSF
    # Neural spline flow (NSF) with 3 sample features and 5 context features
    solver, optimizer, scheduler = get_flow_model(load_model=cfg.use_pretrained)
    knn = load_pickle(file_path=cfg.path_knn)

    solver.train()

    for ep in range(num_epochs):
        t = tqdm(train_loader)
        step = 0
        for batch in t:
            loss = train_step(
                model=solver, batch=batch, optimizer=optimizer, scheduler=scheduler
            )

            bar = {"loss": f"{np.round(loss, 3)}"}
            t.set_postfix(bar, refresh=True)

            step += 1

            if step % cfg.num_steps_eval == 0:
                rand = np.random.randint(low=0, high=len(P_ts), size=cfg.num_eval_size)
                test(
                    robot=panda,
                    P_ts=P_ts[rand],
                    F=F,
                    solver=solver,
                    knn=knn,
                    K=cfg.K,
                    print_report=True,
                )

            if step % cfg.num_steps_save == 0:
                torch.save(solver.state_dict(), cfg.path_solver)


if __name__ == "__main__":
    panda = Robot(verbose=False)

    max_num_epoch = 100
    
    
    sweep_config = {
        "l1": tune.randint(2, 9),   # log transformed with base 2
        "l2": tune.randint(2, 9),   # log transformed with base 2
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(1, 5)    # log transformed with base 2
    }

    mini_train(robot=panda, num_epochs=1, sweep_config=sweep_config)
    
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="ikpflow",
    #     # entity
    #     entity="luca_nthu",
    #     # track hyperparameters and run metadata
    #     config=cfg,
    # )

    # step = 0
    # l2_label = ["mean", "std", "min", "25%", "50%", "75%", "max"]

    # flow.train()
    # for ep in range(cfg.num_epochs):
    #     t = tqdm(train_loader)
    #     for batch in t:
    #         loss = train_step(
    #             model=flow, batch=batch, optimizer=optimizer, scheduler=scheduler
    #         )

    #         bar = {"loss": f"{np.round(loss, 3)}"}
    #         t.set_postfix(bar, refresh=True)

    #         # log metrics to wandb
    #         wandb.log({"loss": np.round(loss, 3)})

    #         step += 1

    #         if step % cfg.num_steps_eval == 0:
    #             df = test(
    #                 robot=panda,
    #                 P_ts=P_ts[: cfg.num_eval_size],
    #                 F=F,
    #                 solver=solver,
    #                 knn=knn,
    #                 K=100,
    #                 print_report=True,
    #             )
    #             l2_val = df.describe().values[1:, 0]
    #             log_info = {}
    #             for l, v in zip(l2_label, l2_val):
    #                 log_info[l] = v
    #             log_info["learning_rate"] = scheduler.get_last_lr()[0]
    #             wandb.log(log_info)

    #         if step % cfg.num_steps_save == 0:
    #             torch.save(flow.state_dict(), cfg.path_solver)

    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()
