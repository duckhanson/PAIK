# Import required packages
from datetime import datetime
import wandb
from utils.robot import get_robot
from train import Trainer
from utils.utils import init_seeds

USE_NSF_CONFIG = True # otherwise, use nf_config
USE_WANDB = True
PATIENCE = 4
POSE_ERR_THRESH = 1.3e-2

nf_config = {
    'name': 'sweep',
    'method': 'bayes',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            # 'values': [900, 1024, 1200]
            'value': 1024
        },
        'subnet_num_layers': {
            # 'values': [3, 4]
            'value': 3
        },
        'num_transforms': {
            'values': [12, 13, 14]  # 6, 8, ..., 16
            # 'value': 12
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 2e-4,
            'max': 6e-4,
            # 'value': 5e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 1e-2,
            'max': 4e-2,
            # 'value': 9.79e-1,
        },
        'decay_step_size': {
            'values': [4e4, 5e4, 6e4],
            # 'value': 2e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 7e-2,
            'max': 9e-2,
            # 'value': 9.79e-1 
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 10
        },
        'model_architecture': {
            'values': ['nf', 'unaf']
        },
        'opt_type': {
            'values': ['adam', 'adamw', 'sgd', 'sgd_nesterov']
        }
    }
}

nsf_config = {
    'name': 'sweep',
    'method': 'bayes',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            # 'values': [900, 1024, 1200]
            'value': 1024
        },
        'subnet_num_layers': {
            # 'values': [3, 4]
            'value': 3
        },
        'num_transforms': {
            'values': [8, 9, 10]  # 6, 8, ..., 16
            # 'value': 12
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 2e-4,
            'max': 6e-4,
            # 'value': 5e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 1e-2,
            'max': 4e-2,
            # 'value': 9.79e-1,
        },
        'decay_step_size': {
            'values': [4e4, 5e4, 6e4],
            # 'value': 2e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 7e-2,
            'max': 9e-2,
            # 'value': 9.79e-1 
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 10
        },
        'model_architecture': {
            'value': 'nsf'
        },
        'opt_type': {
            'values': ['adam', 'adamw', 'sgd', 'sgd_nesterov']
        },
    },
}

sweep_config = nsf_config if USE_NSF_CONFIG else nf_config
    
def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time,
                     notes=f'')

    # note that we define values from `wandb.config`
    # instead of defining hard values
    solver_param = {
        'subnet_width': wandb.config.subnet_width,
        'subnet_num_layers': wandb.config.subnet_num_layers,
        'num_transforms': wandb.config.num_transforms,
        'lr': wandb.config.lr,
        'lr_weight_decay': wandb.config.lr_weight_decay,
        'decay_step_size': wandb.config.decay_step_size,
        'gamma': wandb.config.gamma,
        'batch_size': wandb.config.batch_size,
        'num_epochs': wandb.config.num_epochs,
        'shrink_ratio': 0.31,
        'model_architecture': wandb.config.model_architecture,
        'opt_type': wandb.config.opt_type,
        'ckpt_name': '',
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


if __name__ == '__main__':
    init_seeds(seed=42)
    project_name = 'msik_ikflow_nsf' if USE_NSF_CONFIG else 'msik_ikflow_nf'
    
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=project_name,
                           entity='luca_nthu')
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=20)
    wandb.finish()
    
