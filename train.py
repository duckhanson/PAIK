# Import required packages
# import time
# from os import path

# import numpy as np
# import pandas as pd
import torch
import wandb

# import zuko
# from hnne import HNNE
# from torch import Tensor, nn
from tqdm import tqdm

# from utils.dataset import create_dataset
from utils.model import *
from utils.robot import Robot
from utils.settings import config
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
    x, y = add_small_noise_to_batch(batch)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


if __name__ == "__main__":
    panda = Robot(verbose=False)
    # data generation
    X, y = load_data(robot=panda, num_samples=config.num_train_size)
    # build dimension reduction model
    hnne = get_hnne_model(X, y)
    # get loader
    train_loader = get_loader(X, y, hnne=hnne)
    # get val loader
    val_loader = get_val_loader(robot=panda, hnne=hnne)
    # Build Generative model, NSF
    # Neural spline flow (NSF) with 3 sample features and 5 context features
    flow, optimizer, scheduler = get_flow_model(
        load_model=config.use_pretrained)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ikpflow",
        # entity
        entity="luca_nthu",
        # track hyperparameters and run metadata
        config=config,
    )

    step = 0
    l2_label = ["mean", "std", "min", "25%", "50%", "75%", "max"]

    flow.train()
    for ep in range(config.num_epochs):
        t = tqdm(train_loader)
        for batch in t:
            loss = train_step(
                model=flow, batch=batch, optimizer=optimizer, scheduler=scheduler
            )

            bar = {"loss": f"{np.round(loss, 3)}"}
            t.set_postfix(bar, refresh=True)

            # log metrics to wandb
            wandb.log({"loss": np.round(loss, 3)})

            step += 1

            if step % config.num_steps_eval == 0:
                df, err = test_l2_err(
                    robot=panda,
                    loader=val_loader,
                    model=flow
                )
                l2_val = df.describe().values[1:, 0]
                log_info = {}
                for l, v in zip(l2_label, l2_val):
                    log_info[l] = v
                log_info["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log_info)

            if step % config.num_steps_save == 0:
                torch.save(flow.state_dict(), config.save_path)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
