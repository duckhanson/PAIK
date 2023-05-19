import os
import numpy as np
import math
import pprint
import argparse


import torch
import torch.optim as opt
from torch import nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from utils.csv import Writer
from utils.dataset import create_dataset
from utils.robot import Robot
from utils.settings import param


def compute_log_p_x(y_mb, log_diag_j_mb):
    log_p_y_mb = (
        torch.distributions.Normal(
            torch.zeros_like(y_mb), torch.ones_like(y_mb))
        .log_prob(y_mb)
        .sum(-1)
    )
    return log_p_y_mb + log_diag_j_mb


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def adjust_learning_rate(optimizer, epoch):
    lr = param['lr'] * (0.5 ** (epoch // 110))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class BaseBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(BaseBlock, self).__init__()
        self.l1 = nn.Sequential()
        self.l1.add_module('l1', nn.Linear(in_size, out_size))
        self.l1.add_module('a1', torch.nn.ELU(True))

    def forward(self, x):
        return self.l1(x)


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        blocks = [
            # kwargs["input_length"]
            # kwargs["DoF"] + kwargs['goal_length']
            BaseBlock(kwargs['input_length'], kwargs['layer_length']),
            BaseBlock(kwargs['layer_length'], kwargs['layer_length']),
            BaseBlock(kwargs['layer_length'], int(kwargs['layer_length']/2)),
            BaseBlock(int(kwargs['layer_length']/2),
                      int(kwargs['layer_length']/4)),
            nn.Linear(int(kwargs['layer_length']/4), kwargs['layer_length']),
        ]
        self.l1 = nn.Sequential(*blocks)

        self.mu = nn.Linear(kwargs['layer_length'], kwargs['output_length'])
        self.log_sigma = nn.Linear(
            kwargs['layer_length'], kwargs['output_length'])

    def __sample_latent(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        std_z = torch.normal(mean=0, std=1, size=sigma.size(
        ), requires_grad=False, dtype=torch.float32, device=log_sigma.device)
        self.z_mean = mu
        self.z_sigma = sigma

        # Reparameterization trick
        return mu + sigma * std_z

    def sample(self, x):
        mu, log_sigma = self.forward(x)
        return self.__sample_latent(mu, log_sigma)

    def forward(self, x):
        x = self.l1(x)
        return self.mu(x), self.log_sigma(x)


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        blocks = [
            # kwargs["input_length"]
            # kwargs["latent_length"] + kwargs['goal_length']
            BaseBlock(kwargs['input_length'], int(kwargs['layer_length'] / 4)),
            BaseBlock(int(kwargs['layer_length'] / 4),
                      int(kwargs['layer_length'] / 2)),
            BaseBlock(int(kwargs['layer_length'] / 2),
                      int(kwargs['layer_length'] / 2)),
            BaseBlock(int(kwargs['layer_length'] / 2),
                      int(kwargs['layer_length'] / 4)),
            nn.Linear(int(kwargs['layer_length'] / 4), kwargs['output_length'])
        ]
        self.l1 = nn.Sequential(*blocks)

    def forward(self, x):
        return self.l1(x)


class VAE(nn.Module):
    def __init__(self, x_length, goal_length, layer_length, latent_length, device):
        super(VAE, self).__init__()
        encoder = Encoder(input_length=x_length + goal_length,
                          layer_length=layer_length, output_length=latent_length)
        decoder = Decoder(input_length=latent_length + goal_length,
                          layer_length=layer_length, output_length=x_length)
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def __sample_latent(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        std_z = torch.normal(mean=0, std=1, size=sigma.size(
        ), requires_grad=False, dtype=torch.float32, device=self.device)
        self.z_mean = mu
        self.z_sigma = sigma

        # Reparameterization trick
        return mu + sigma * std_z

    def forward(self, state, labels):
        mu, log_sigma = self.encoder(state)
        z = self.__sample_latent(mu, log_sigma)
        z = torch.cat((z, labels), dim=1)
        return self.decoder(z)


class VAE_R(nn.Module):
    def __init__(self, x_length, goal_length, layer_length, latent_length, device):
        super(VAE_R, self).__init__()
        encoder = Encoder(input_length=x_length,
                          layer_length=layer_length, output_length=latent_length)
        decoder = Decoder(input_length=latent_length + goal_length,
                          layer_length=layer_length, output_length=x_length)
        encoder.to(device)
        decoder.to(device)

        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, q, labels):
        z = self.encoder.sample(q)
        z = torch.cat((z, labels), dim=1)
        return self.decoder(z)


class BaseModel:
    '''
    The base model for creating a training model VAE and providing methods to train and sample.
    '''

    def __init__(self, verbose: bool, x_length: int, goal_length: int,
                 latent_length: int, layer_length: int, device: str,
                 drop: bool = False):
        '''
        Create a training model VAE and provide methods to train and sample.

        Parameters
        ----------
        verbose : bool
            print information or not.
        x_length : int
            model inputs, the data part of VAE. 
        goal_length : int
            model inputs, the condition part of VAE.
        latent_length : int
            model latent layer length.
        layer_length : int
            model inner layer length.
        device : str
            device acclerates model computing.
        drop : bool, optional
            drop saved model weights, by default False
        '''
        self.model = VAE(x_length=x_length,
                         goal_length=goal_length,
                         latent_length=latent_length,
                         layer_length=layer_length,
                         device=device
                         )
        self.verbose = verbose
        self.device = device
        self.panda = Robot(verbose=False)

        self.no_improvement = 0
        self.loss_min = np.inf
        self.loss_log = []
        self.start_epoch = 0

        self.ckpt_dir_path = None
        self.best_model_path = None
        self.loss_image_path = None

    def plot_loss(self):
        plt.plot(np.array(self.loss_log), 'r')
        plt.savefig(self.loss_image_path)

    def load(self):
        if os.path.exists(self.best_model_path):
            state = torch.load(self.best_model_path)
            # print(state)
            self.loss_min = state['loss']
            self.loss_log = state['loss_log']
            self.start_epoch = state['epoch']
            self.model.encoder.load_state_dict(state['encoder'])
            self.model.decoder.load_state_dict(state['decoder'])

            print(
                f"Load from {self.best_model_path} successfully, epoch: {self.start_epoch}, loss: {self.loss_min}")
        else:
            print("Not exist pretrained weights.")
            if not os.path.exists(self.ckpt_dir_path):
                os.mkdir(self.ckpt_dir_path)

    def __drop(self):
        os.remove(self.best_model_path)

    def drop_if_exists(self):
        exist = os.path.exists(self.best_model_path)
        if exist:
            print(f"Drop best model from {self.best_model_path}")
            self.__drop()

    def save(self, epoch, loss):
        self.loss_log.append(loss)

        # Detect nan
        if loss == np.nan:
            raise ValueError("Exist nan, needs smaller lr.")

        # Save best model
        elif loss < self.loss_min:
            self.no_improvement = 0
            self.loss_min = loss
            state = {
                'loss': loss,
                'loss_log': self.loss_log,
                'epoch': epoch,
                'encoder': self.model.encoder.state_dict(),
                'decoder': self.model.decoder.state_dict(),
            }
            torch.save(state, self.best_model_path)

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

    def sample(self, n_samples: int, n_poses: int, acc_ratio: float = 0.7):
        if n_samples == 0:
            return

        # load best model
        self.load()
        self.model.to(self.device)
        writer = Writer(tb_name=self.tb_name,
                        verbose=self.verbose,
                        write_period=self.write_period,
                        denormalize=True,
                        mean=self.ds.true_mean,
                        stddev=self.ds.true_stddev)
        # Create dataset, dataloader
        if n_poses == -1:
            n_poses = self.ds.__len__()

        loader = self.ds.create_loader(shuffle=True,
                                       batch_size=n_poses)
        # sample
        with torch.no_grad():
            for q_ten, goal_ten in loader:
                q_np = q_ten.numpy()
                goal_np = goal_ten.numpy()

                q_with_goal = torch.cat((q_ten, goal_ten), dim=1)

                goal_ten = goal_ten.to(self.device)
                q_with_goal = q_with_goal.to(self.device)

                for _ in trange(n_samples, ncols=80):
                    q_hat = self.model(q_with_goal, goal_ten)
                    q_hat = q_hat.cpu().detach().numpy()
                    batch_info = np.column_stack((q_hat, goal_np))
                    writer.add_batch(batch_info)
                break
        writer.save()

    def sample_ee_path(self, ee_path: np.array, n_samples: int = 3):
        if n_samples == 0:
            return

        # load best model
        self.load()
        self.model.to(self.device)

        # Create dataset, dataloader
        loader = self.ds.create_loader(shuffle=True,
                                       batch_size=param['batch_size'])
        
        writer = Writer(tb_name=self.tb_name,
                        verbose=self.verbose,
                        write_period=self.write_period,
                        denormalize=True,
                        mean=self.ds.true_mean,
                        stddev=self.ds.true_stddev)
        goal_mean = self.ds.true_mean[param['panda_dof']:]
        goal_stddev = self.ds.true_stddev[param['panda_dof']:]
        

        # sample
        t = tqdm(ee_path, ncols=80)
        with torch.no_grad():
            for goal in t:
                goal = (goal - goal_mean) / goal_stddev
                goal_batch = np.broadcast_to(
                    goal, (loader.batch_size, len(goal)))

                goal = torch.tensor(goal_batch, dtype=torch.float32, device=self.device)

                for _ in range(n_samples):
                    for q, _ in loader:
                        q_with_goal = torch.cat((q, goal), dim=1)
                        q_hat = self.model(q_with_goal, goal)
                        q_hat = q_hat.cpu().detach().numpy()
                        goal_np = goal.cpu().detach().numpy()

                        batch_info = np.column_stack((q_hat, goal_np))
                        write, total = writer.add_batch(batch_info)

                        bar = {
                            "write/total": f"{write}/{total}",
                        }
                        t.set_postfix(bar, refresh=True)
        writer.save()

    def train(self, min_epochs: int, max_epochs: int):
        # TODO
        '''
        _summary_

        Parameters
        ----------
        min_epochs : int
            serves as a lower bound of minmum training epochs.
        max_epochs : int
            serves as a upper bound of maxmum training epochs.
        '''
        if max_epochs == 0:
            return
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        self.load()
        self.model.to(self.device)

        criterion = nn.MSELoss()
        optimizer = opt.Adam(self.model.parameters(), lr=param['lr'])

        loss_log = []

        t = trange(self.start_epoch + 1, self.max_epochs, ncols=80)

        for epoch in t:
            adjust_learning_rate(optimizer, epoch)
            epoch_loss = []
            for q, goal in self.loader:
                q = q.to(self.device)
                goal = goal.to(self.device)
                q_with_goal = torch.cat((q, goal), dim=1)

                optimizer.zero_grad()
                q_hat = self.model(q_with_goal, goal)
                KL_loss = latent_loss(self.model.z_mean, self.model.z_sigma)
                loss = criterion(q_hat, q) + KL_loss * param['kl_ratio']
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(loss)

            epoch_loss = torch.stack(epoch_loss).mean()

            bar = {
                "epoch": epoch,
                "epoch_loss": "{:.5f}".format(epoch_loss.item()),
            }
            t.set_postfix(bar, refresh=False)

            try:
                self.save(epoch, epoch_loss.item())
            except Exception as e:
                print(e)
                break
        self.plot_loss()


class TrainingModel(BaseModel):
    '''
    Training model class. Only supports tb_name="psik" or "esik".
    '''

    def __init__(self, tb_name: str, verbose: bool, drop: bool = False,
                 write_period: int = 10_000):
        '''
        Training model class. Only supports tb_name="psik" or "esik".

        Parameters
        ----------
        tb_name : str
            Support psik, esik datasets. From dataset gets generated joint configurations.
        verbose : bool
            print information or not.
        drop : bool, optional
            drop saved model weights, by default False

        Raises
        ------
        ValueError
            if tb_name is not supported.
        '''

        if tb_name == 'psik':
            ds_name = 'degree'
            goal_length = param['goal_length']
            layer_length = param['layer_length']
        elif tb_name == 'esik':
            ds_name = 'local'
            goal_length = param['goal_length'] + 1
            layer_length = int(param['layer_length']/4)
        else:
            raise ValueError(f"Not support dataset {tb_name}")

        super().__init__(verbose=verbose,
                         x_length=param['panda_dof'],
                         goal_length=goal_length,
                         latent_length=param['latent_length'],
                         layer_length=layer_length,
                         device=param['device'], drop=drop)

        self.tb_name = tb_name
        self.verbose = verbose
        self.write_period = write_period

        ds = create_dataset(tb_name=ds_name, verbose=False,
                            enable_normalize=True)

        self.ds = ds

        self.loader = ds.create_loader(shuffle=True,
                                       batch_size=param['batch_size'])

        self.ckpt_dir_path = f"{param['weight_dir']}/{param[f'{tb_name}_table']}"
        self.best_model_path = f"{self.ckpt_dir_path}/best.ckpt"
        self.loss_image_path = f"{param['image_dir']}/{param[f'{tb_name}_table']}"

        if drop:
            self.drop_if_exists()
