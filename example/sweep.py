# Import required packages
from datetime import datetime
from tabulate import tabulate
import wandb
from paik.settings import (
    PANDA_PAIK,
    FETCH_PAIK,
    FETCH_ARM_PAIK,
    ATLAS_ARM_PAIK,
    ATLAS_WAIST_ARM_PAIK,
    BAXTER_ARM_PAIK,

    PANDA_NSF,
    FETCH_NSF,
    FETCH_ARM_NSF,
    ATLAS_ARM_NSF,
    ATLAS_WAIST_ARM_NSF,
    BAXTER_ARM_NSF,
)
from paik.train import Trainer

WORK_DIR = "/home/luca/paik"
# please change to your own project name
WANDB_PROJECT_NAME = f"FETCH_NSF FINCH"
SOLVER_PARAM = FETCH_NSF # [CHANGE THIS]
WANDB_ENTITY = "luca_nthu"  # please change to your own entity name
PATIENCE = 10
EXPERMENT_COUNT = 10
NUM_EPOCHS = 60
# USE_NSF_ONLY = False
ENABLE_LOAD_MODEL = False
USE_DIMENSION_REDUCTION = False


def get_range(left_bound, right_bound, scale):
    return [i * scale for i in range(left_bound, right_bound)]


sweep_config = {
    "name": "sweep",
    "method": "random",
    "metric": {"name": "position_errors", "goal": "minimize"},
    "parameters": {
        "num_transforms": {"values": get_range(8, 9, 1)},
        "lr": {"values": get_range(50, 80, 1e-5)}, # 40 - 80
        "lr_weight_decay": {"values": get_range(16, 20, 1e-3)},
        "gamma": {"values": get_range(85, 87, 1e-3)},
        "noise_esp": {"values": get_range(31, 33, 1e-4)},
        "noise_esp_decay": {"values": get_range(98, 99, 1e-2)},
        "num_bins": {"values": get_range(10, 11, 1)},
        "base_std": {"values": get_range(40, 60, 1e-2)},
        "lr_beta_l": {"values": get_range(93, 94, 1e-2)},
        "lr_beta_h": {"values": get_range(94, 95, 1e-2)},
    },
}


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    wandb.init(name=begin_time)

    solver_param = SOLVER_PARAM
    solver_param.num_transforms = wandb.config.num_transforms
    solver_param.lr = wandb.config.lr
    solver_param.lr_weight_decay = wandb.config.lr_weight_decay
    solver_param.gamma = wandb.config.gamma
    solver_param.noise_esp = wandb.config.noise_esp
    solver_param.noise_esp_decay = wandb.config.noise_esp_decay
    solver_param.num_bins = wandb.config.num_bins
    solver_param.base_std = wandb.config.base_std
    solver_param.lr_beta = (wandb.config.lr_beta_l, wandb.config.lr_beta_h)
    solver_param.use_dimension_reduction = USE_DIMENSION_REDUCTION  # type: ignore
    solver_param.enable_load_model = ENABLE_LOAD_MODEL  # type: ignore
    solver_param.workdir = WORK_DIR
    
    # train nsf model, check use_nsf_only is True
    assert solver_param.use_nsf_only, "use_nsf_only must be True"

    trainer = Trainer(solver_param=solver_param)

    (
        avg_l2_errs,
        avg_ang_errs,
        avg_inference_time,  # type: ignore
        _,
    ) = trainer.evaluate_ikp_iterative(num_poses=100, num_sols=1000, verbose=False)

    trainer.mini_train(
        num_epochs=NUM_EPOCHS,
        batch_size=solver_param.batch_size,
        begin_time=begin_time,
        patience=PATIENCE,
        num_eval_poses=1000,
        num_eval_sols=100,
    )


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY
    )
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=EXPERMENT_COUNT)
    wandb.finish()
