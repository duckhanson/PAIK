# Import required packages
from datetime import datetime
import wandb
from utils.robot import get_robot
from train import Trainer
from utils.utils import init_seeds


USE_WANDB = True
PATIENCE = 4
POSE_ERR_THRESH = 1.1e-2

sweep_config = {
    'name': 'sweep',
    'method': 'random',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            'values': [900, 1024, 1200]
            # 'value': 1024
        },
        'subnet_num_layers': {
            # 'values': [3, 4]
            'value': 3
        },
        'num_transforms': {
            'values': [7, 8, 9]  # 6, 8, ..., 16
            # 'value': 11
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 2e-4,
            'max': 9e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 1e-2,
            'max': 9e-2,
        },
        'decay_step_size': {
            'values': [2e4, 4e4, 6e4],
            # 'value': 2e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 5e-2,
            'max': 9.9e-2,
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 20
        }
    },
}
    
def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time,
                     notes=f'smallest but 1 cm model')

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
        'shrink_ratio': 0.61,
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
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=f'msik_2.4M_klampt',
                           entity='luca_nthu')
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=15)
    wandb.finish()
    
