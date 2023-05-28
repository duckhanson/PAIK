# Import required packages
from os import path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
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


def get_flow_model(load_model=True):
    """
    Return nsf model and optimizer

    :return: (nsf, AdamW, StepLR)
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
    
    flow = get_sflow_model(flow)

    if load_model and path.exists(path=config.save_path):
        try:
            flow.load_state_dict(state_dict=torch.load(config.save_path))
            print(f'Model load successfully from {config.save_path}')
        except:
            print(f'Load err, assuming you use different architecture.')
    else:
        print('Create a new model and start training.')

    # Train to maximize the log-likelihood
    optimizer = AdamW(flow.parameters(), lr=config.lr, weight_decay=config.lr_weight_decay)
    scheduler = StepLR(optimizer, step_size=config.decay_step_size, gamma=config.decay_gamma)
    
    return flow, optimizer, scheduler

def get_iflow_model(flow: NSF, init_sample: torch.Tensor, shrink_ratio: float = 0.01) -> FlowModule:
    """
    sampling from initial samples as search space centers
    seach shrink_ratio region

    :param flow: _description_
    :type flow: NSF
    :param init_sample: _description_
    :type init_sample: torch.Tensor
    :param shrink_ratio: _description_, defaults to 0.01
    :type shrink_ratio: float, optional
    :return: _description_
    :rtype: FlowModule
    """
    iflow = FlowModule(
        transforms=flow.transforms, 
        base= Unconditional(
                DiagNormal,
                torch.zeros((config.dof,)) + init_sample,
                torch.ones((config.dof,)) * shrink_ratio,
                buffer=True,
        ))
    
    iflow.to(config.device)
    
    return iflow
    

def get_sflow_model(flow: NSF, shrink_ratio: float = config.shrink_ratio):
    """
    shrink normal distribution model

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
    sflow = FlowModule(
        transforms=flow.transforms, 
        base= Unconditional(
                DiagNormal,
                torch.zeros((config.dof,)),
                torch.ones((config.dof,))*config.shrink_ratio,
                buffer=True,
        ))
    
    sflow.to(config.device)
    
    return sflow

def get_nflow_model(flow: NSF):
    """
    _summary_

    :param flow: _description_
    :type flow: NSF
    :return: _description_
    :rtype: _type_
    """
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



def get_hnne_model(X: np.array, y: np.array, save_path: str = config.hnne_save_path):
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
    
    suc_load = False
    if path.exists(path=save_path):
        try:
            hnne = HNNE.load(path=save_path)
            print(f'hnne load successfully from {save_path}')
            X_trans = load_numpy(file_path=config.x_trans_train_path)
            suc_load = True
        except:
            print(f'hnne load err, assuming you use different architecture.')
        
    if not suc_load:
        hnne = HNNE(dim=config.reduced_dim, ann_threshold=config.num_neighbors)
        X_trans = hnne.fit_transform(X=X, dim=config.reduced_dim, verbose=True)
        hnne.save(path=save_path)
        save_numpy(file_path=config.x_trans_train_path, arr=X_trans)
    
    return hnne
    
