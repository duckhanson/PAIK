import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from utils.settings import config
from utils.robot import Robot

def load_data(robot, num_samples:int = config.num_samples, return_ee: bool = True, generate_new: bool = False):
    X = []
    if not generate_new:    
        # data generation
        X = load_numpy(file_path=config.x_data_path)
        y = load_numpy(file_path=config.y_data_path)

    if len(X) != num_samples:
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
    
def add_small_noise_to_batch(batch, esp: float = config.noise_esp, step: int = 0, eval: bool = False):
    x, y = batch
    if eval or step < 1_0000:
        std = torch.zeros((x.shape[0], 1)).to(config.device)
        y = torch.column_stack((y, std))
    else:
        c = torch.rand((x.shape[0], 1)).to(config.device)
        y = torch.column_stack((y, c))
        noise = torch.normal(mean=torch.zeros_like(input=x), std=esp*torch.repeat_interleave(input=c, repeats=x.shape[1], dim=1))
        x = x + noise
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
    df = pd.DataFrame(np.column_stack((errs, log_probs)), columns=['l2_err', 'log_prob'])
    return df, errs.mean()

def save_show_pose_data(config, num_data, num_samples, model=flow):
    """
    _summary_
    example of use: save_show_pose_data(config, num_data=5, num_samples=10, model=nflow)
    :param config: _description_
    :type config: _type_
    :param num_data: _description_
    :type num_data: _type_
    :param num_samples: _description_
    :type num_samples: _type_
    :param model: _description_, defaults to flow
    :type model: _type_, optional
    """
    batch = next(iter(loader))
    x, y = add_small_noise_to_batch(batch, eval=True)
    assert num_data < len(x)
    
    x_hats = np.array([])
    pidxs = np.array([])
    errs = np.array([])
    log_probs = np.array([])
    rand = np.random.randint(low=0, high=len(x), size=num_data)
    
    for nd in rand:
        x_hat = model(y[nd]).sample((num_samples,))
        log_prob = model(y[nd]).log_prob(x_hat)
        
        x_hat = x_hat.detach().cpu().numpy()
        log_prob = -log_prob.detach().cpu().numpy()
        target = y[nd].detach().cpu().numpy()
        # ee_pos = ee_pos * (ds.targets_max - ds.targets_min) + ds.targets_min
        ee_pos = target[:3]
        
        for q in x_hat:
            err = panda.dist_fk(q=q, ee_pos=ee_pos)
            errs = np.concatenate((errs, [err]))
        x_hats = np.concatenate((x_hats, x_hat.reshape(-1)))
        pidx = target[3:-1]
        pidx = np.tile(pidx, (num_samples, 1))

        pidxs = np.concatenate((pidxs, pidx.reshape(-1)))
        log_probs = np.concatenate((log_probs, log_prob))

    x_hats = x_hats.reshape((-1, panda.dof))
    pidxs = pidxs.reshape((len(x_hats), -1))
    

    save_numpy(config.show_pose_features_path, x_hats)
    save_numpy(config.show_pose_pidxs_path, pidxs)
    save_numpy(config.show_pose_errs_path, errs)
    save_numpy(config.show_pose_log_probs_path, log_probs)
    
    print('Save pose successfully')

def inside_same_pidx():
    """
    _summary_
    example of use: 
    save_show_pose_data(config, num_data=5, num_samples=10, model=nflow)
    inside_same_pidx()
    :raises ValueError: _description_
    """
    x_hats = load_numpy(file_path=config.show_pose_features_path)
    pidxs = load_numpy(file_path=config.show_pose_pidxs_path)
    
    if len(x_hats) == 0:
        raise ValueError("lack show pose data") 
    
    pre_pidx = None
    qs = np.array([])
    for q, pidx in zip(x_hats, pidxs):
        if pre_pidx is None or np.array_equal(pre_pidx, pidx):
            qs = np.concatenate((qs, q))
        else:
            break
        pre_pidx = pidx
    qs = qs.reshape((-1, panda.dof))
    for q in qs:
        panda.plot(q, q)