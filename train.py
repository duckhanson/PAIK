# Import required packages
from os import path 
import wandb
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from tqdm import tqdm, trange
import zuko
from zuko.flows import Distribution, NSF
from zuko.distributions import DiagNormal, BoxUniform, Minimum
from zuko.flows import DistributionModule, FlowModule, Unconditional
from hnne import HNNE

from utils.settings import config
from utils.utils import *
from utils.model import *
from utils.robot import Robot
from utils.dataset import create_dataset

def train_step(model, batch, optimizer, scheduler):
    """
    _summary_

    :param model: _description_
    :type model: _type_
    :param batch: _description_
    :type batch: _type_
    :param optimizer: _description_
    :type optimizer: _type_
    :param scheduler: _description_
    :type scheduler: _type_
    :return: _description_
    :rtype: _type_
    """
    x, y = add_small_noise_to_batch(batch)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return loss.item()

if __name__ == '__main__':
    panda = Robot(verbose=False)
    # data generation
    X, y = load_data(robot=panda)
    # build dimension reduction model
    hnne, ds, loader = get_hnne_model(X, y)
    # Build Generative model, NSF
    # Neural spline flow (NSF) with 3 sample features and 5 context features
    flow, optimizer, scheduler = get_flow_model(load_model=config.use_pretrained)
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ikpflow",
        
        # track hyperparameters and run metadata
        config=config
    )
    
    step = 0
    l2_label = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

    flow.train()
    for ep in range(config.num_epochs):
        t = tqdm(loader)
        for batch in t:
            loss = train_step(model=flow, batch=batch, optimizer=optimizer, scheduler=scheduler)
            
            bar = {"loss": f"{np.round(loss, 3)}"}
            t.set_postfix(bar, refresh=True)
            
            # log metrics to wandb
            wandb.log({"loss": np.round(loss, 3)})

            step += 1
            if step % config.num_steps_save == 0:
                torch.save(flow.state_dict(), config.save_path)
                df, err = test_l2_err(config, robot=panda, loader=loader, model=flow)
                l2_val = df.describe().values[1:, 0]
                log_info = {}
                for l, v in zip(l2_label, l2_val):
                    log_info[l] = v
                log_info['learning_rate'] = scheduler.get_last_lr()[0]
                wandb.log(log_info)
    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
    
