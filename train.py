# Import required packages
from datetime import datetime
from jrl.robot import Robot
import torch 
import numpy as np

import math
from tqdm import tqdm
from pprint import pprint
import wandb
from utils.robot import get_robot
from utils.settings import config as cfg
from utils.utils import init_seeds, EarlyStopping, get_train_loader, train_step

from utils.solver import Solver, DEFAULT_SOLVER_PARAM_M7, DEFAULT_SOLVER_PARAM_M3

USE_WANDB = False
# NUM_RECORD_STEPS = 14e3
PATIENCE = 4    
POSE_ERR_THRESH = 7e-3

class Trainer(Solver):
    def __init__(self, robot: Robot, solver_param: dict) -> None:
        super().__init__(robot, solver_param)
        
    def mini_train(self, begin_time, num_epochs, batch_size=128, use_wandb=True, patience=4, pose_err_thres=1e-2, num_eval_poses=100, num_eval_sols=100) -> None:
        init_seeds()
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # data generation
        assert self._device == 'cuda', "device should be cuda"

        train_loader = get_train_loader(J=self._J_tr,
                                        P=self._P_tr,
                                        F=self._F, # type: ignore
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
                                scheduler=self._scheduler)
                batch_loss[step] = loss
                bar = {"loss": f"{np.round(loss, 3)}"}
                t.set_postfix(bar, refresh=True)
                if np.isnan(loss):
                    print(f"Early stopping ({loss} is nan)")
                    break
                step += 1
            
            avg_position_error, avg_orientation_error = self.random_evaluation(num_poses=num_eval_poses, num_sols=num_eval_sols) # type: ignore
            
            if use_wandb:
                wandb.log({
                    'ep': ep,
                    'position_errors': avg_position_error,
                    'orientation_errors': avg_orientation_error,
                    'train_loss': batch_loss.mean(),
                })
            else:
                pprint({
                    'ep': ep,
                    'position_errors': avg_position_error,
                    'orientation_errors': avg_orientation_error,
                    'train_loss': batch_loss.mean(),
                })

            if np.isnan(avg_position_error) or avg_position_error > 1e-1:
                print(f"Early stopping ({avg_position_error} > 1e-1)")
                break
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(avg_position_error, self._solver)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
                    
        if avg_position_error < pose_err_thres: # type: ignore
            torch.save({
                'solver': self._solver.state_dict(),
                'opt': self._optimizer.state_dict(),
                }, f'{cfg.weight_dir}{begin_time}.pth')
            
        del train_loader
        print("Finished Training")


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    if USE_WANDB:   
        wandb.init(name=begin_time,
                         notes=f'r=0')
    
    solver_param = {
        'subnet_width': 1024,
        'subnet_num_layers': 3,
        'num_transforms': 8,
        'lr': 1.3e-4,
        'lr_weight_decay': 3.1e-2,
        'decay_step_size': 6e4,
        'gamma': 9e-2,
        'shrink_ratio': 0.61,
        'batch_size': 128,
        'num_epochs': 15,
        'ckpt_name': '1019-1842',
        'nmr': (7, 7, 1),
    }
    
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
