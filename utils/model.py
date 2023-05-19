# Import required packages
from os import path
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
from utils.utils import load_numpy, save_numpy
from utils.robot import Robot
from utils.dataset import create_dataset


def get_flow_model():
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW)
    :rtype: tuple
    """
    # Build Generative model, NSF
    # Neural spline flow (NSF) with inputs 7 features and 3 + 4 + 1 context
    flow = NSF(features=config.num_features, 
            context=config.num_conditions, 
            transforms=config.num_transforms, 
            randperm=True, 
            activation=config.activation, 
            hidden_features=config.subnet_shape).to(config.device)

    # flow.load_state_dict(state_dict=torch.load(config.save_path))

    if path.exists(path=config.save_path):
        flow.load_state_dict(state_dict=torch.load(config.save_path))
        print(f'Model load successfully from {config.save_path}')

    # Train to maximize the log-likelihood
    optimizer = AdamW(flow.parameters(), lr=config.lr, weight_decay=config.lr_decay)
    
    return flow, optimizer

def get_nflow_model(flow: NSF):
    nflow = FlowModule(
        transforms=flow.transforms, 
        base= Unconditional(
                BoxUniform,
                -torch.ones((config.dof,))*config.shrink_ratio,
                torch.ones((config.dof,))*config.shrink_ratio,
                buffer=True,
        ))
    
    nflow.to(config.device)
    
    return nflow

def get_hnne_model(X: np.array, y: np.array, return_ds: bool = True):
    """
    Return hnne model

    :param X: feature / joint config
    :type X: np.array
    :param y: target / position
    :type y: np.array
    :param return_ds: return dataset and loader, defaults to True
    :type return_ds: bool, optional
    :return: hnne model (builded), if return_ds, then return (hnne, ds, loader).
    :rtype: tuple
    """
    # build dimension reduction model
    hnne = HNNE(dim=config.reduced_dim, ann_threshold=config.num_neighbors)
    X_transformed = hnne.fit_transform(X=X, dim=config.reduced_dim, verbose=True)
    
    if return_ds:
        y = np.column_stack((y, X_transformed))
        ds = create_dataset(features=X, targets=y, enable_normalize=False)
        loader = ds.create_loader(shuffle=True, batch_size=config.batch_size)
    
        return hnne, ds, loader
    
    return hnne
    
