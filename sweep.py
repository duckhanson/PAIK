# Import required packages
from paik.solver import DEFAULT_SOLVER_PARAM_M7_NORM
import torch
from datetime import datetime
import wandb
from paik.settings import SolverConfig
from train import Trainer
from paik.utils import init_seeds

USE_WANDB = True
PATIENCE = 5
POSE_ERR_THRESH = 4e-3
EXPERMENT_COUNT = 10
NUM_EPOCHS = 25


sweep_config = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"name": "position_errors", "goal": "minimize"},
    "parameters": {
        "num_transforms": {
            # "values": [7, 8]  # 6, 8, ..., 16
            "value": 8
        },
        "lr": {
            # a flat distribution between 0 and 0.1
            "values": [i * 1e-7 for i in range(7, 310)]
            # 'value': 5e-4,
        },
        "lr_weight_decay": {
            # a flat distribution between 0 and 0.1
            "values": [i * 1e-3 for i in range(20, 29)]
            # 'value': 9.79e-1,
        },
        "decay_step_size": {
            "values": [4e4, 5e4, 6e4],
            # "value": 4e4
        },
        "gamma": {
            "values": [0.84, 0.85, 0.86]
            # "value": 8.6e-1
        },
        "noise_esp": {
            "values": [i * 1e-4 for i in range(20, 31)]
            # "distribution": "q_uniform",
            # "q": 1e-4,
            # "min": 2.0e-3,
            # "max": 3.0e-3,
        },
        "noise_esp_decay": {
            "values": [0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
            # 'value': 9.79e-1
        },
        "shrink_ratio": {
            "values": [i * 1e-2 for i in range(51, 77)]
            # "value": False
        },
    },
}


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time, notes=f"")

    solver_param = DEFAULT_SOLVER_PARAM_M7_NORM
    solver_param.lr = wandb.config.lr
    solver_param.lr_weight_decay = wandb.config.lr_weight_decay
    solver_param.decay_step_size = wandb.config.decay_step_size
    solver_param.noise_esp = wandb.config.noise_esp
    solver_param.noise_esp_decay = wandb.config.noise_esp_decay

    trainer = Trainer(solver_param=solver_param)

    print(trainer.random_evaluation(num_poses=100, num_sols=100))

    trainer.mini_train(
        num_epochs=solver_param.num_epochs,
        batch_size=solver_param.batch_size,
        begin_time=begin_time,
        use_wandb=USE_WANDB,
        patience=PATIENCE,
        pose_err_thres=POSE_ERR_THRESH,
        num_eval_poses=100,
        num_eval_sols=100,
    )


if __name__ == "__main__":
    init_seeds(seed=42)
    project_name = "msik_ikflow_nsf_norm"

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name, entity="luca_nthu")
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=EXPERMENT_COUNT)
    wandb.finish()
