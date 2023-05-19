import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from utils.settings import param
from utils.dataset import create_dataset
from utils.robot import Robot
from utils.csv import Writer
from utils.model import BaseModel

def load_numpy(file_path):
    if os.path.exists(path=file_path):
        return np.load(file_path)
    else:
        print(f'{file_path}: file not exist and return empty np array.')
        return np.array([])

def creat_unique_goal_cluster(ds):
    def get_uniq_goals(ds):
        df = pd.DataFrame(ds.goal.cpu())
        uni_goals = df.drop_duplicates()
        
        uni_ids = uni_goals.index
        uni_ids = uni_ids.append(other=pd.Index([df.__len__()]))
        
        uni_goals = uni_goals.to_numpy()
        uni_ids = uni_ids.to_numpy()
        return uni_goals, uni_ids
    
    uni_goals, uni_ids = get_uniq_goals(ds)

    joint_configs = ds.x.cpu().numpy()
    
    cluster = {}
    for i in range(len(uni_ids)-1):
        # print(f"cluster begins: {uni_ids[i]}, ends: {uni_ids[i+1]}, tuple: {tuple(uni_goals[i])}", end=' ')
        cluster[str((tuple(uni_goals[i])))] = joint_configs[uni_ids[i]: uni_ids[i+1]]
        # print(f"size: {len(cluster[(tuple(uni_goals[i]))])}")
    # print(cluster)
    return cluster, uni_goals

class Model_Saver:
    def __init__(self, model_name, min_epochs, max_epochs):
        ckpt_dir_path = f"{param['weight_dir']}/{model_name}"

        if not os.path.exists(ckpt_dir_path):
            os.mkdir(ckpt_dir_path)

        model_path = f"{ckpt_dir_path}/best.ckpt"
        if not os.path.exists(model_path):
            print(f"Not find pre-trained weights @ {model_path}.")

        self.model_path = model_path
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.start_epoch = 0
        self.loss_image_path = f"{param['image_dir']}/{model_name}"
        self.loss_log = []
        self.loss_min = float('inf')
        self.no_improvement = 0

    def plot_loss(self):
        plt.plot(np.array(self.loss_log), 'r')
        plt.savefig(self.loss_image_path)

    def save(self, model=None, vae=None, loss_list: list = None):
        if loss_list is not None:
            self.loss_log = loss_list

        if model is not None:
            state = {
                'loss_log': self.loss_log,
                'model': model.state_dict(),
            }
        elif vae is not None:
            state = {
                'loss_log': self.loss_log,
                'encoder': vae.encoder.state_dict(),
                'decoder': vae.decoder.state_dict(),
            }
        else:
            raise NotImplementedError("Input should have either model or vae")

        torch.save(state, self.model_path)

    def load(self, model):
        state = torch.load(self.model_path)
        self.loss_log = state['loss_log']
        self.plot_loss()

        model.load_state_dict(state['model'])
        self.start_epoch = len(self.loss_log)
        print(
            f"Load from {self.model_path} successfully, epoch: {self.start_epoch}")

    def drop_if_exists(self):
        exist = os.path.exists(self.model_path)
        if exist:
            print(f"Drop best model from {self.model_path}")
            os.remove(self.model_path)

    def save_epoch(self, epoch, loss, model=None, vae=None):
        self.loss_log.append(loss)

        # Detect nan
        if loss == np.nan:
            raise ValueError("Exist nan, needs smaller lr.")

        # Save best model
        elif loss < self.loss_min:
            self.no_improvement = 0
            self.loss_min = loss
            self.save(model=model, vae=vae)

        # Early stop
        else:
            self.no_improvement += 1
            if self.no_improvement > param['patience']:
                raise ValueError(f"Early Stop, total_epochs={epoch}.")

        # min epochs
        if epoch < self.min_epochs:
            self.no_improvement = 0

        # max epochs
        if epoch > self.max_epochs:
            raise Exception(f"Touch max epochs, total_epochs={epoch}.")


def print_model_param(model):
    for name, para in model.named_parameters():
        print("-"*20)
        print(f"name: {name}")
        print("values: ")
        print(para)


def quality_check(tb_name: str, shuffle: bool, n_samples: int, z_ver: bool = False):
    # TODO
    '''
    _summary_

    Parameters
    ----------
    tb_name : str
        _description_
    shuffle : bool
        _description_
    n_samples : int
        _description_
    z_ver : bool, optional
        _description_, by default False
    '''
    panda = Robot(verbose=False)

    ds = create_dataset(tb_name=tb_name, verbose=False,
                        enable_normalize=False, z_ver=z_ver)

    loss = 0
    n_cnt = min(n_samples, len(ds.x))

    for i, row in enumerate(ds.df.to_numpy()):
        if i == n_samples:
            break

        q = row[:param['panda_dof']]
        goal = row[param['panda_dof']:]
        ee_pos = goal[:param['goal_length']]
        diff = panda.dist_fk(q, ee_pos)
        loss += diff

    print(f"{tb_name}: {round(loss/n_cnt, 8)}, units: (m)")


# def num_ik(tb_name: str, threshold: float = 1e-4,
#            write_period: int = 1000, verbose: bool = False):
#     '''
#     Use Numeric Inverse Kinematics to improve generated joint configuartion

#     Parameters
#     ----------
#     tb_name : str
#         Support psik, esik datasets. From dataset gets generated joint configurations.

#     Returns
#     -------
#     utils.Writer class
#         a writer class provides a save path to check computed quality.

#     Raises
#     ------
#     ValueError
#         if tb_name is not supported.
#     '''

#     panda = Robot(verbose=False)

#     if tb_name == 'psik':
#         writer_name = 'num_p'
#     elif tb_name == 'esik':
#         writer_name = 'num_e'
#     else:
#         raise ValueError(f"Not support dataset {tb_name}")

#     path_dict, ee_path = path_generator.read()
    
#     record_dict = {}
#     def find_T(ee_pos):
#         T = record_dict.get(tuple(ee_pos), None)
#         if T is None:
#             shift_path = ee_path - ee_pos
#             min_sum = float('inf')
#             min_idx = -1
#             for idx, row in enumerate(shift_path):
#                 row_sum = np.sum(np.abs(row))
#                 if min_sum > row_sum:
#                     min_sum = row_sum
#                     min_idx = idx
                    
#             T = path_dict[tuple(list(ee_path[min_idx]))]
#             record_dict[tuple(ee_pos)] = T
#         return T
    
#     ds = create_dataset(tb_name=tb_name, verbose=False, enable_normalize=False)
#     writer = Writer(tb_name=writer_name, verbose=verbose,
#                     write_period=write_period, denormalize=False)

#     t = tqdm(ds, ncols=80)
#     curr = 0
#     suc_cnt = 0
#     for q, goal in t:
#         curr += 1
#         q_hat = q.cpu().numpy()
#         goal = goal.cpu().numpy()
#         ee_pos = np.copy(goal[:param['goal_length']])
#         T = find_T(ee_pos)
#         q_num, success = panda.inverse_kinematics(T, q0=q_hat)
#         if success:
#             suc_cnt += 1
#         dist = panda.dist_fk(q_num, ee_pos)
#         if dist > threshold:
#             print(dist)
#             continue
#         row_info = np.concatenate((q_num, goal))
#         writer.add_poses(row_info)
#         bar = {
#             "(acc/curr)": f"{writer.poses_cnt}/{curr}",
#             "succ": suc_cnt,
#         }
#         t.set_postfix(bar, refresh=False)

#     writer.save()
#     print(f"total number of q: {writer.poses_cnt}/{len(ds.x)}")
#     writer.cat_old()

#     return writer


def ee_path(from_pos: np.array, to_pos: np.array, t: int, verbose: bool):
    '''
    Given from_pos and to_pos generate an end effector position path.


    Use example

        >>> ee_path(from_pos=[100, 0], to_pos=[0, 100], t=20)


    Parameters
    ----------
    from_pos : np.array
        start end effector (ee) position
    to_pos : np.array
        end ee position
    t : int
        number of time slices
    verbose : bool
        print out step information

    Returns
    -------
    np.array
        an array of ee postion from from_pos to to_pos.

    Raises
    ------
    ValueError
        from_pos and to_pos should have the same size, e.g. vec of 2 dim or 3 dim.
    ValueError
        t should be a positive integer.
    '''
    if len(from_pos) != len(to_pos):
        raise ValueError(
            'from_pos and to_pos should have the same size, e.g. vec of 2 dim or 3 dim.')

    if t <= 0:
        raise ValueError('t should be a positive integer.')

    from_pos = np.array(from_pos, dtype=np.float32)
    to_pos = np.array(to_pos, dtype=np.float32)

    diff = (to_pos - from_pos)/t

    if verbose:
        print(f'step diff = {diff}')

    curr_pos = from_pos
    path = []
    for _ in range(t+1):
        path.append(curr_pos)
        curr_pos = curr_pos + diff

    path = np.array(path)

    if verbose:
        print(f'path: \n {path}')

    return path


class WriteLatentZ(BaseModel):
    def __init__(self, verbose: bool, tb_name: str, write_period: int, device: str):
        # TODO
        '''
        _summary_

        Parameters
        ----------
        verbose : bool
            _description_
        tb_name : str
            _description_
        write_period : int
            _description_
        device : str
            _description_
        '''
        super().__init__(verbose=verbose,
                         x_length=param['panda_dof'],
                         goal_length=param['goal_length'],
                         latent_length=param['latent_length'],
                         layer_length=param['layer_length'],
                         device=device)

        ds = create_dataset(tb_name=tb_name, verbose=False,
                            enable_normalize=True)
        self.ds = ds
        self.loader = ds.create_loader(shuffle=False,
                                       batch_size=param['batch_size'])

        self.writer = Writer(tb_name='local', verbose=True, write_period=write_period,
                             denormalize=True, mean=ds.true_mean, stddev=ds.true_stddev,
                             z_ver=True)

        self.ckpt_dir_path = f"{param['weight_dir']}/{param['psik_table']}"
        self.best_model_path = f"{self.ckpt_dir_path}/best.ckpt"
        self.loss_image_path = f"{param['image_dir']}/{param['psik_table']}"

    def generator(self):
        # Set in evaluate mode
        for q, goal in self.loader:
            # Xs, Ps, Js: batch(X), batch(P), batch(J)
            q_np = q.numpy()
            goal_np = goal.numpy()

            goal = goal[:, :param['goal_length']]
            q_with_goal = torch.cat((q, goal), dim=1)

            q_with_goal = q_with_goal.to(self.device)
            goal = goal.to(self.device)

            yield q_np, goal, goal_np, q_with_goal

    def fit(self):
        # load best model
        self.load()
        self.model.to(self.device)
        # Create database table, checking "IF EXISTS" inside the function.
        self.writer.drop_if_exists()
        # sample
        with torch.no_grad():
            for i, (q_np, goal, goal_np, q_with_goal) in enumerate(self.generator()):
                q_hat = self.model(q_with_goal, goal)
                z = self.model.z_mean

                z = z.cpu().detach().numpy()
                batch_info = np.column_stack((q_np, goal_np, z))
                self.writer.add_batch(batch_info)
        self.writer.save()
