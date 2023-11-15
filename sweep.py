# Import required packages
import torch
from datetime import datetime
import wandb
from paik.model import get_robot
from train import Trainer
from paik.utils import init_seeds

USE_WANDB = True
PATIENCE = 4
POSE_ERR_THRESH = 7.3e-3
EXPERMENT_COUNT = 20


sweep_config = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"name": "position_errors", "goal": "minimize"},
    "parameters": {
        "subnet_width": {
            # 'values': [1024, 1150]
            "value": 1024
        },
        "subnet_num_layers": {
            # 'values': [3, 4]
            "value": 3
        },
        "num_transforms": {
            # 'values': [8]  # 6, 8, ..., 16
            "value": 8
        },
        "lr": {
            # a flat distribution between 0 and 0.1
            "distribution": "q_uniform",
            "q": 1e-5,
            "min": 3.0e-4,
            "max": 4.8e-4,
            # 'value': 5e-4,
        },
        "lr_weight_decay": {
            # a flat distribution between 0 and 0.1
            "distribution": "q_uniform",
            "q": 1e-3,
            "min": 1e-2,
            "max": 3e-2,
            # 'value': 9.79e-1,
        },
        "decay_step_size": {
            # 'values': [4e4, 5e4, 6e4],
            "value": 4e4
        },
        "gamma": {
            "distribution": "q_uniform",
            "q": 1e-3,
            "min": 8.4e-2,
            "max": 8.6e-2,
            # 'value': 9.79e-1
        },
        "batch_size": {"value": 128},
        "num_epochs": {"value": 20},
        "model_architecture": {"value": "nsf"},
        "noise_esp": {
            "distribution": "q_uniform",
            "q": 1e-4,
            "min": 1.7e-3,
            "max": 3.3e-3,
            # 'value': 9.79e-1
        },
        "noise_esp_decay": {
            "distribution": "q_uniform",
            "q": 1e-2,
            "min": 9.4e-1,
            "max": 9.9e-1,
            # 'value': 9.79e-1
        },
        "opt_type": {
            # 'values': ['adam', 'adamw', 'sgd', 'sgd_nesterov']
            "value": "adamw"
        },
        "sche_type": {
            # 'values': ['step', 'plateau'], # cos bad
            "value": "plateau"
        },
        "random_perm": {
            "values": [False, True]
            # 'value': False
        },
        "shrink_ratio": {
            "distribution": "q_uniform",
            "q": 1e-2,
            "min": 3.1e-1,
            "max": 6.0e-1,
        },
    },
}


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time, notes=f"")

    # note that we define values from `wandb.config`
    # instead of defining hard values
    solver_param = {
        "subnet_width": wandb.config.subnet_width,
        "subnet_num_layers": wandb.config.subnet_num_layers,
        "num_transforms": wandb.config.num_transforms,
        "lr": wandb.config.lr,
        "lr_weight_decay": wandb.config.lr_weight_decay,
        "decay_step_size": wandb.config.decay_step_size,
        "gamma": wandb.config.gamma,
        "batch_size": wandb.config.batch_size,
        "num_epochs": wandb.config.num_epochs,
        "shrink_ratio": wandb.config.shrink_ratio,
        "model_architecture": wandb.config.model_architecture,
        "random_perm": wandb.config.random_perm,
        "enable_load_model": False,
        "noise_esp": wandb.config.noise_esp,
        "noise_esp_decay": wandb.config.noise_esp_decay,
        "opt_type": wandb.config.opt_type,
        "sche_type": wandb.config.sche_type,
        "ckpt_name": "",
        "nmr": (7, 7, 1),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = Trainer(robot=get_robot(), solver_param=solver_param)

    trainer.mini_train(
        num_epochs=solver_param["num_epochs"],
        batch_size=solver_param["batch_size"],
        begin_time=begin_time,
        use_wandb=USE_WANDB,
        patience=PATIENCE,
        pose_err_thres=POSE_ERR_THRESH,
        num_eval_poses=100,
        num_eval_sols=100,
    )


if __name__ == "__main__":
    init_seeds(seed=42)
    project_name = "msik_ikflow_nsf"

    sweep_id = wandb.sweep(
        sweep=sweep_config, project=project_name, entity="luca_nthu")
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=EXPERMENT_COUNT)
    wandb.finish()
