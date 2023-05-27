import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from utils.settings import config

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
    
def test_l2_err(config, robot, loader, model, num_data = config.num_test_data, num_samples = config.num_test_samples, inference: bool=False):
    errs = np.zeros(shape=(num_data))
    log_probs = np.zeros(shape=(num_data))
    time_diff = np.zeros(shape=(num_data))
    
    for i in range(num_data):
        er, lp, dt = __test_one_l2_err(config, robot, loader, model, num_samples=num_samples, inference=inference)
        errs[i] = er
        log_probs[i] = lp
        time_diff[i] = dt
    
    if inference:
        df = pd.DataFrame(data=np.column_stack((errs, time_diff)), columns=['l2_err', f'time_diff(per {num_samples})'])
    else:
        df = pd.DataFrame(data=np.column_stack((errs, log_probs, time_diff)), columns=['l2_err', 'log_prob', f'time_diff(per {num_samples})'])
        
    return df, errs.mean()

def __test_one_l2_err(config, robot, loader, model, num_samples: int, inference: bool):
    batch = next(iter(loader))
    x, y = add_small_noise_to_batch(batch, eval=True)

    errs = np.zeros((num_samples,))
    log_probs = np.zeros((num_samples,))
    rand = np.random.randint(low=0, high=len(x), size=1)
    
    # assuming rand is a number
    time_begin = time.time()
    model.eval()
    x_hat = model(y[rand]).sample((num_samples,))
    time_diff = round(time.time() - time_begin, 2)
        
    qs = x_hat.detach().cpu().numpy()
    ee_pos = y[rand].detach().cpu().numpy()
    ee_pos = ee_pos[0, :3]
    
    step = 0
    for q in qs:
        errs[step] = robot.l2_err_func(q=q, ee_pos=ee_pos)
        step += 1
    
    if not inference:
        log_prob = model(y[rand]).log_prob(x_hat)
        log_prob = -log_prob.detach().cpu().numpy()
        
    return errs.mean(), log_probs.mean(), time_diff
    

def save_show_pose_data(config, num_data, num_samples, model, robot):
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
            err = robot.l2_err_func(q=q, ee_pos=ee_pos)
            errs = np.concatenate((errs, [err]))
        x_hats = np.concatenate((x_hats, x_hat.reshape(-1)))
        pidx = target[3:-1]
        pidx = np.tile(pidx, (num_samples, 1))

        pidxs = np.concatenate((pidxs, pidx.reshape(-1)))
        log_probs = np.concatenate((log_probs, log_prob))

    x_hats = x_hats.reshape((-1, robot.dof))
    pidxs = pidxs.reshape((len(x_hats), -1))
    

    save_numpy(config.show_pose_features_path, x_hats)
    save_numpy(config.show_pose_pidxs_path, pidxs)
    save_numpy(config.show_pose_errs_path, errs)
    save_numpy(config.show_pose_log_probs_path, log_probs)
    
    print('Save pose successfully')

def inside_same_pidx(robot):
    """
    _summary_
    example of use: 
    save_show_pose_data(config, num_data=5, num_samples=10, model=nflow)
    inside_same_pidx(robot)
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
    qs = qs.reshape((-1, robot.dof))
    for q in qs:
        robot.plot(q=q, p=q)

def sample_jtraj(path, pidx, model, robot):
    """
    _summary_
    example of use: 
    df, qs = sample_jtraj(ee_traj, px, nflow)
    :param path: _description_
    :type path: _type_
    :param pidx: _description_
    :type pidx: _type_
    :param model: _description_
    :type model: _type_
    :return: _description_
    :rtype: _type_
    """
    path_len = len(path)
    pidx = np.tile(pidx, (path_len,1))
    cstd = np.zeros((path_len,))
    
    y = np.column_stack((path, pidx, cstd))
    y = torch.tensor(data=y, device='cuda', dtype=torch.float32)
    
    errs = np.zeros((len(path),))
    log_probs = np.zeros((len(path),))
    
    model.eval()
    x_hat = model(y).sample((1,))
    log_prob = model(y).log_prob(x_hat)
    
    x_hat = x_hat.detach().cpu().numpy()[0]
    log_prob = -log_prob.detach().cpu().numpy()[0]

    step = 0
    ang_errs = np.zeros_like(errs)
    for q, lp, ee_pos in zip(x_hat, log_prob, path):
        if step != 0:
            ang_errs[step] = abs(np.sum(x_hat[step-1] - q))
        errs[step] = robot.l2_err_func(q=q, ee_pos=ee_pos)
        log_probs[step] = lp     
        step += 1
    ang_errs[0] = np.average(ang_errs[1:])
    ang_errs = np.rad2deg(ang_errs)
    df = pd.DataFrame(np.column_stack((errs, log_probs, ang_errs)), columns=['l2_err', 'log_prob', 'ang_errs(sum)'])
    qs = x_hat
    return df, qs

def sample_ee_traj(robot, load_time: str = '') -> str:
    """
    _summary_
    example of use 
    
    for generate
    traj_dir = sample_ee_traj(robot=panda, load_time=')
    
    for demo
    traj_dir = sample_ee_traj(robot=panda, load_time='05232300')
    
    :param robot: _description_
    :type robot: _type_
    :param load_time: _description_, defaults to ''
    :type load_time: str, optional
    :return: _description_
    :rtype: str
    """
    if load_time == '':
        traj_dir = config.traj_dir + datetime.now().strftime('%m%d%H%M') + '/'
    else:
        traj_dir = config.traj_dir + load_time + '/'
        
    ee_traj_path = traj_dir + 'ee_traj.npy'
    q_traj_path = traj_dir + 'q_traj.npy'
    
    if load_time == '' or not os.path.exists(path=q_traj_path):
        ee_traj, q_traj = robot.path_generate_via_stable_joint_traj(dist_ratio=0.9, t=20)
        save_numpy(file_path=ee_traj_path, arr=ee_traj)
        save_numpy(file_path=q_traj_path, arr=q_traj)
    
    if os.path.exists(path=traj_dir):
        print(f"{traj_dir} load successfully.")
        
    return traj_dir

def generate_traj_via_model(robot, traj_dir: str, hnne=None, model=None, num_traj: int = 3, outliner_thres: float = 5e-2, enable_regenerate: bool = False) -> None:
    """
    _summary_
        for generate
        generate_traj_via_model(hnne=hnne, num_traj=3, model=nflow, robot=panda, traj_dir=traj_dir)
        
        only for demo
        generate_traj_via_model(hnne=None, num_traj=3, model=None, robot=panda, traj_dir=traj_dir)
    
    :param robot: _description_
    :type robot: _type_
    :param traj_dir: _description_
    :type traj_dir: str
    :param hnne: _description_, defaults to None
    :type hnne: _type_, optional
    :param model: _description_, defaults to None
    :type model: _type_, optional
    :param num_traj: _description_, defaults to 3
    :type num_traj: int, optional
    :param outliner_ccthres: _description_, defaults to 5e-2
    :type outliner_ccthres: float, optional
    :param enable_regenerate: _description_, defaults to False
    :type enable_regenerate: bool, optional
    """
    def load_and_plot(exp_traj_path: str, ee_path: np.array):
        if os.path.exists(path=exp_traj_path):
            err = np.zeros((100,))
            qs = load_numpy(file_path=exp_traj_path)
            for i in range(len(qs)):
                err[i] = robot.l2_err_func(q=qs[i], ee_pos=ee_traj[i])
            outliner = np.where(err > outliner_thres)
            print(outliner)
            print(err[outliner])
            print(np.sum(err))
            robot.plot(qs=qs)
        else:
            print(f'{exp_traj_path} does not exist !')
    
    already_exists = True
    exp_path = lambda idx: traj_dir + f'exp_{i}.npy'
    for i in range(num_traj):
        if not os.path.exists(path=exp_path(i)):
            already_exists = False
            break
        
    ee_traj = load_numpy(file_path=traj_dir + 'ee_traj.npy')
    q_traj = load_numpy(file_path=traj_dir + 'q_traj.npy')
    
    if enable_regenerate or not already_exists and hnne is not None and model is not None:
        rand = np.random.randint(low=0, high=len(q_traj), size=num_traj)
        pidx = hnne.transform(X=q_traj[rand])
        print(pidx)
    
        for i, px in enumerate(pidx):
            df, qs = sample_jtraj(ee_traj, px, model, robot)
            print(df.describe())
            save_numpy(file_path=exp_path(i), arr=qs)
    
    for i in range(num_traj):
        load_and_plot(exp_traj_path=exp_path(i), ee_path=ee_traj)
            

def create_robot_dirs() -> None:
    """
    _summary_
    """
    for dp in config.dir_paths:
        if not os.path.exists(path=dp):
            os.makedirs(name=dp)
            print(f"Create {dp}")


def remove_empty_robot_dirs() -> None:
    """
    _summary_
    """
    for dp in config.dir_paths:
        walk = list(os.walk(dp))
        for p, _, _ in walk[::-1]:
            if os.path.exists(p) and len(os.listdir(p)) == 0:
                os.removedirs(p)
                print(f"Delete {p}")
                
    
        
    