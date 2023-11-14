# Import required packages
from datetime import datetime
from jrl.robot import Robot
import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint
import wandb
from utils.model import get_robot
from utils.settings import config as cfg
from utils.utils import init_seeds
from utils.dataset import get_train_loader

from utils.solver import Solver, DEFAULT_SOLVER_PARAM_M7, DEFAULT_SOLVER_PARAM_M3

USE_WANDB = False
# NUM_RECORD_STEPS = 14e3
PATIENCE = 4
POSE_ERR_THRESH = 7e-3


def train_step(model, batch, optimizer, noise_esp: float, std_scale: float):
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
    x, y = add_noise(batch, esp=noise_esp, std_scale=std_scale)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class Trainer(Solver):
    def __init__(self, robot: Robot, solver_param: dict) -> None:
        super().__init__(robot, solver_param)
        self.__noise_esp = self.param['noise_esp']
        self.__noise_esp_decay = self.param['noise_esp_decay']  # 0.8 - 0.9
        self.__std_scale = 1 / self.__noise_esp

    def __update_noise_esp(self, epoch: int):
        self.__noise_esp = self.param['noise_esp'] * \
            (self.__noise_esp_decay ** epoch)

    def mini_train(self, begin_time, num_epochs, batch_size=128, use_wandb=True, patience=4, pose_err_thres=1e-2, num_eval_poses=100, num_eval_sols=100) -> None:
        init_seeds()
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # data generation
        assert self._device == 'cuda', "device should be cuda"

        train_loader = get_train_loader(J=self._J_tr,
                                        P=self._P_tr,
                                        F=self._F,  # type: ignore
                                        device=self._device,
                                        batch_size=batch_size)

        self._solver.train()

        for ep in range(num_epochs):
            t = tqdm(train_loader)
            batch_loss = np.zeros((len(train_loader)))
            step = 0
            for batch in t:
                loss = train_step(model=self._solver,
                                  batch=batch,
                                  optimizer=self._optimizer,
                                  noise_esp=self.__noise_esp,
                                  std_scale=self.__std_scale)
                batch_loss[step] = loss
                bar = {"loss": f"{np.round(loss, 3)}"}
                t.set_postfix(bar, refresh=True)
                if np.isnan(loss):
                    print(f"Early stopping ({loss} is nan)")
                    break
                step += 1

                if self.param['sche_type'] == 'step':
                    self._scheduler.step()  # type: ignore
            self.__update_noise_esp(ep)

            self.shrink_ratio = 0.25
            print(
                f"using shrink_ratio: {self.shrink_ratio} (fixed), where original shrink_ratio: {self.param['shrink_ratio']} (training)")
            avg_position_error, avg_orientation_error = self.random_evaluation(
                num_poses=num_eval_poses, num_sols=num_eval_sols)  # type: ignore
            self.shrink_ratio = self.param['shrink_ratio']  # type: ignore

            if self.param['sche_type'] == 'plateau':
                self._scheduler.step(avg_position_error)  # type: ignore
            elif self.param['sche_type'] == 'cos':
                self._scheduler.step()  # type: ignore

            log_info = {
                'lr': self._optimizer.param_groups[0]['lr'],  # type: ignore
                'position_errors': avg_position_error,
                'orientation_errors': avg_orientation_error,
                'train_loss': batch_loss.mean(),
                'noise_esp': self.__noise_esp,
            }

            if use_wandb:
                wandb.log(log_info)
            else:
                pprint(log_info)

            if np.isnan(avg_position_error) or avg_position_error > 1e-1:
                print(f"Early stopping ({avg_position_error} > 1e-1)")
                break

            if ep > 14 and avg_position_error > 1.5e-2:
                print(f"Early stopping ({avg_position_error} > 1e-2)")
                break

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(avg_position_error, self._solver)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if avg_position_error < pose_err_thres:  # type: ignore
            torch.save({
                'solver': self._solver.state_dict(),
                'opt': self._optimizer.state_dict(),
            }, f'{cfg.weight_dir}{begin_time}.pth')

        del train_loader
        print("Finished Training")

def add_noise(batch, esp: float, std_scale: float):
    J, C = batch
    if esp < 1e-9:
        std = torch.zeros((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, std))
    else:
        # softflow implementation
        s = esp * torch.rand((C.shape[0], 1)).to(C.device)
        C = torch.column_stack((C, std_scale * s))
        noise = torch.normal(
            mean=torch.zeros_like(input=J),
            std=torch.repeat_interleave(input=s, repeats=J.shape[1], dim=1),
        )
        J = J + noise
    return J, C

class EarlyStopping:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, enable_save=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.enable_save = enable_save
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {score}, best: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.enable_save:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
            torch.save(model.state_dict(), self.path)
        else:
            if self.verbose:
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).')
        self.val_loss_min = val_loss

def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    if USE_WANDB:
        wandb.init(name=begin_time,
                   notes=f'r=0')

    solver_param = DEFAULT_SOLVER_PARAM_M7

    trainer = Trainer(robot=get_robot(),
                      solver_param=solver_param)

    trainer.mini_train(num_epochs=solver_param['num_epochs'],
                       batch_size=solver_param['batch_size'],
                       begin_time=begin_time,
                       use_wandb=USE_WANDB,
                       patience=PATIENCE,
                       pose_err_thres=POSE_ERR_THRESH,
                       num_eval_poses=100,
                       num_eval_sols=100)

    if USE_WANDB:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
    else:
        pprint(f"Finish job {begin_time}")

if __name__ == "__main__":
    init_seeds()
    main()