# Import required packages
from datetime import datetime
from tabulate import tabulate
import wandb
from paik.settings import get_config
from paik.train import Trainer

WORK_DIR = "/home/luca/paik"
# please change to your own project name
SOVLER_ARCH = "paik"
ROBOT_NAME = "panda"

WANDB_PROJECT_NAME = f"{ROBOT_NAME} {SOVLER_ARCH} FINCH"
SOLVER_PARAM = get_config(SOVLER_ARCH, ROBOT_NAME)
WANDB_ENTITY = "luca_nthu"  # please change to your own entity name
PATIENCE = 10
EXPERMENT_COUNT = 10
NUM_EPOCHS = 60

def get_range(left_bound, right_bound, scale):
    return [i * scale for i in range(left_bound, right_bound)]


sweep_config = {
    "name": "sweep",
    "method": "random",
    "metric": {"name": "position_errors", "goal": "minimize"},
    "parameters": {
        "num_transforms": {"value": 8},
        "num_bins": {"value": 10},
        "lr_beta_l": {"value": 93*1e-2},
        "lr_beta_h": {"value": 94*1e-2},
        "lr": {"values": get_range(50, 80, 1e-5)},  # 40 - 80
        "lr_weight_decay": {"values": get_range(16, 20, 1e-3)},
        "noise_esp": {"value": 31*1e-4},
        "noise_esp_decay": {"value": 98*1e-2},
        "gamma": {"values": get_range(85, 87, 1e-3)},
        "base_std": {"values": get_range(40, 60, 1e-2)},
    },
}


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    wandb.init(name=begin_time)

    solver_param = SOLVER_PARAM
    solver_param.model_architecture = SOVLER_ARCH
    solver_param.num_transforms = wandb.config.num_transforms
    solver_param.lr = wandb.config.lr
    solver_param.lr_weight_decay = wandb.config.lr_weight_decay
    solver_param.gamma = wandb.config.gamma
    solver_param.noise_esp = wandb.config.noise_esp
    solver_param.noise_esp_decay = wandb.config.noise_esp_decay
    solver_param.num_bins = wandb.config.num_bins
    solver_param.base_std = wandb.config.base_std
    solver_param.lr_beta = (wandb.config.lr_beta_l, wandb.config.lr_beta_h)
    solver_param.workdir = WORK_DIR

    # train nsf model, check use_nsf_only is True
    # assert solver_param.use_nsf_only, "use_nsf_only must be True"

    trainer = Trainer(solver_param=solver_param)

    trainer.solver.random_ikp(num_poses=100, num_sols=1000, verbose=False)

    trainer.mini_train(
        num_epochs=NUM_EPOCHS,
        batch_size=solver_param.batch_size,
        begin_time=begin_time,
        patience=PATIENCE,
        num_eval_poses=500,
        num_eval_sols=200,
    )


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY
    )
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=EXPERMENT_COUNT)
    wandb.finish()
