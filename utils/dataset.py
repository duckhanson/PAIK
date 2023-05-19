import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from utils.settings import param
from utils.csv import Reader


def create_dataset(tb_name: str = None, 
                   verbose: bool = False,
                   features: np.array = None,
                   targets: np.array = None,
                   enable_normalize: bool = False, 
                   z_ver: bool = False, 
                   device: str = param['device'], 
                   data_get_item: bool = False):
    if tb_name is None and (features is None or targets is None):
        raise ValueError("tb_name is None and (features or targets are None)")
    
    if features is not None and targets is not None:
        return CustomDataset(features, targets, device=device, enable_normalize=enable_normalize)
    
    if tb_name == 'num_p':
        return BaseDataset(tb_name=tb_name, verbose=verbose,
                           enable_normalize=enable_normalize,
                           sort_by=['ee_px', 'ee_py', 'ee_pz'],
                           device=device,
                           data_get_item=data_get_item)
    elif tb_name == 'num_e' or tb_name == 'local':
        return BaseDataset(tb_name=tb_name, verbose=verbose,
                           enable_normalize=enable_normalize,
                           sort_by=['ee_px', 'ee_py', 'ee_pz', 'ml'],
                           device=device,
                           data_get_item=data_get_item)
    else:
        return BaseDataset(tb_name=tb_name, verbose=verbose,
                           enable_normalize=enable_normalize, z_ver=z_ver,
                           device=device,
                           data_get_item=data_get_item)

class CustomDataset(Dataset):
    def __init__(self, features, targets, device, enable_normalize):
        if len(features) != len(targets):
            raise ValueError("features and targets should have the same shape[0].")
        
        features = np.array(features)
        targets = np.array(targets)
        self.df = pd.DataFrame(data=np.column_stack((features, targets)))
        self.device = device
        self.features = torch.tensor(features, device=device, dtype=torch.float32)
        self.targets = torch.tensor(targets, device=device, dtype=torch.float32)
        if enable_normalize:
            self.normalize()
        
    def normalize(self):
        r_min = self.df.min(axis=0).to_numpy()
        r_max = self.df.max(axis=0).to_numpy()
        
        # use min max normalize
        self.features_min = torch.tensor(r_min[:self.features.shape[-1]], device=self.device, dtype=torch.float32)
        self.features_max = torch.tensor(r_max[:self.features.shape[-1]], device=self.device, dtype=torch.float32)
        self.targets_min = torch.tensor(r_min[self.features.shape[-1]:], device=self.device, dtype=torch.float32)
        self.targets_max = torch.tensor(r_max[self.features.shape[-1]:], device=self.device, dtype=torch.float32)
        
        self.features = (self.features - self.features_min) / (self.features_max - self.features_min)
        self.targets = (self.targets - self.targets_min) / (self.targets_max - self.targets_min)
        
                
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, id):
        return self.features[id], self.targets[id]
    
    def create_loader(self, shuffle, batch_size):
        return DataLoader(self, 
                          batch_size=batch_size,
                          shuffle=shuffle, 
                          drop_last=True)


class BaseDataset(Reader):
    def __init__(self, tb_name, verbose, enable_normalize, z_ver: bool = False,
                 sort_by: list = [], device: str = 'cpu', data_get_item: bool = False):
        super().__init__(tb_name, verbose, enable_normalize, z_ver, sort_by)
        self.__dof = 7
        self.device = device
        self.data_get_item = data_get_item
        self.__prepare_data()

    def __prepare_data(self):
        """ Prepare data for training
        In our case:
        Parm:
            x: the joint configs
            goal: the position (and oriantation)
        """
        self.read_data()
        self.data = torch.tensor(self.data, dtype=torch.float32, device=self.device)

        x = self.data[:, :self.__dof]
        goal = self.data[:, self.__dof:]

        if self.z_ver:
            z = goal[:, -param['latent_length']:]
            goal = goal[:, :-param['latent_length']]
            self.z = z.clone().detach()

        self.x = x.clone().detach()
        self.goal = goal.clone().detach()
        # self.n_dims = param['latent_length'] + param['goal_length']
        
        # del self.data

    def __len__(self):
        return len(self.goal)

    def __getitem__(self, id):
        if self.z_ver:
            return self.x[id], self.goal[id], self.z[id]
        elif self.data_get_item:
            return self.data[id]
        else:
            return self.x[id], self.goal[id]
        
    def random_sample(self, size):
        rand = torch.randint(0, self.__len__(), size=(size,))
        return self[rand]

    def create_loader(self, shuffle, batch_size):
        return DataLoader(self, 
                          batch_size=batch_size,
                          shuffle=shuffle, 
                          drop_last=True)

    def create_train_val_loader(self, train_ratio, batch_size):
        ## Dataset ##
        

        train_len = int(self.__len__() * train_ratio)
        val_len = self.__len__() - train_len

        train_set, val_set = random_split(self, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, drop_last=True,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, drop_last=True,
                                pin_memory=True)
        return train_loader, val_loader
