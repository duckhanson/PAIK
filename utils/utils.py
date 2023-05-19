import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from utils.settings import config
from utils.robot import Robot

def load_data(robot, num_samples:int = 100_0000, return_ee: bool = True, generate_new: bool = False):
    X = []
    if not generate_new:    
        # data generation
        X = load_numpy(file_path=config.x_data_path)
        y = load_numpy(file_path=config.y_data_path)

    if len(X) < num_samples:
        X, y = robot.random_sample_joint_config(num_samples=num_samples, return_ee=True)
        save_numpy(file_path=config.x_data_path, arr=X)
        save_numpy(file_path=config.y_data_path, arr=y)

    return X, y

def load_numpy(file_path):
    if os.path.exists(path=file_path):
        return np.load(file_path)
    else:
        print(f'{file_path}: file not exist and return empty np array.')
        return np.array([])
    
def save_numpy(file_path, arr):
    dir_path = os.path.dirname(p=file_path)
    if not os.path.exists(path=dir_path):
        print(f'mkdir {dir_path}')
        os.mkdir(path=dir_path)
    np.save(file_path, arr)
    
def add_small_noise_to_batch(batch, esp: float = config.noise_esp, eval: bool = False):
    x, y = batch
    if eval:
        std = torch.zeros((x.shape[0], 1)).to(config.device)
        y = torch.column_stack((y, std))
    else:
        std = torch.rand((x.shape[0], 1)).to(config.device)
        y = torch.column_stack((y, std))
        noise = torch.normal(mean=torch.zeros_like(input=x), std=torch.repeat_interleave(input=std, repeats=x.shape[1], dim=1))
        x = x + esp * noise
    return x, y
    
def test_l2_err(config, robot, loader, model, step=None):
    num_data, num_samples = config.num_test_data, config.num_test_samples
    batch = next(iter(loader))
    x, y = add_small_noise_to_batch(batch, eval=True)
    assert num_data < len(x)

    errs = np.zeros((num_data*num_samples,))
    log_probs = np.zeros((num_data*num_samples,))
    rand = np.random.randint(low=0, high=len(x), size=num_data)
    
    step = 0
    for nd in rand:
        x_hat = model(y[nd]).sample((num_samples,))
        log_prob = model(y[nd]).log_prob(x_hat)
        
        x_hat = x_hat.detach().cpu().numpy()
        log_prob = -log_prob.detach().cpu().numpy()
        ee_pos = y[nd].detach().cpu().numpy()
        # ee_pos = ee_pos * (ds.targets_max - ds.targets_min) + ds.targets_min
        ee_pos = ee_pos[:3]
        
        for q, lp in zip(x_hat, log_prob):
            errs[step] = robot.dist_fk(q=q, ee_pos=ee_pos)
            log_probs[step] = lp     
            step += 1
    print(f'step={step}')
    df = pd.DataFrame(np.column_stack((errs, log_probs)), columns=['l2_err', 'log_prob'])
    return df, errs.mean()