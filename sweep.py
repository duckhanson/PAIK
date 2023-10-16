# Import required packages
from datetime import datetime
import wandb
from train import mini_train
from utils.utils import init_seeds


USE_WANDB = True
# NUM_RECORD_STEPS = 14e3
PATIENCE = 3
POSE_ERR_THRESH = 1e-2

sweep_config = {
    'name': 'sweep',
    'method': 'random',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            'values': [1500, 1600]
            # 'value': 1600
        },
        'subnet_num_layers': {
            # 'values': [3, 4]
            'value': 3
        },
        'num_transforms': {
            'values': [15, 16]  # 6, 8, ..., 16
            # 'value': 16
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 1e-4,
            'max': 5e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 1e-2,
            'max': 4e-2,
        },
        'decay_step_size': {
            'values': [5e4, 6e4, 7e4],
            # 'value': 4e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 9e-2,
            'max': 9.5e-2,
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 15
        }
    },
}
    
def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time,
                     notes=f'25M')

    # note that we define values from `wandb.config`
    # instead of defining hard values
    config = {
        'subnet_width': wandb.config.subnet_width,
        'subnet_num_layers': wandb.config.subnet_num_layers,
        'num_transforms': wandb.config.num_transforms,
        'lr': wandb.config.lr,
        'lr_weight_decay': wandb.config.lr_weight_decay,
        'decay_step_size': wandb.config.decay_step_size,
        'gamma': wandb.config.gamma,
        'batch_size': wandb.config.batch_size,
        'num_epochs': wandb.config.num_epochs,
    }

    mini_train(config=config,
               begin_time=begin_time,
               use_wandb=USE_WANDB,
               patience=PATIENCE,
               pose_err_thres=POSE_ERR_THRESH)


if __name__ == '__main__':
    init_seeds(seed=42)
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project=f'msik_2.5M_JP_dim_red',
                           entity='luca_nthu')
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=2)
    wandb.finish()
    
