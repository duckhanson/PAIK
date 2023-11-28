# Import required packages
from paik.solver import DEFAULT_SOLVER_PARAM_M7_NORM
from datetime import datetime
import wandb
from train import Trainer
from paik.utils import init_seeds

USE_WANDB = True
PATIENCE = 5
POSE_ERR_THRESH = 5.5e-3
EXPERMENT_COUNT = 20
NUM_EPOCHS = 25
ENABLE_LODE_MODEL = False


def get_range(left_bound, right_bound, scale):
    return [i * scale for i in range(left_bound, right_bound)]


sweep_config = {
    "name": "sweep",
    "method": "bayes",
    "metric": {"name": "position_errors", "goal": "minimize"},
    "parameters": {
        "lr": {"values": get_range(30, 68, 1e-5)},
        "lr_weight_decay": {"values": get_range(10, 30, 1e-3)},
        "gamma": {"values": get_range(84, 87, 1e-3)},
        "noise_esp": {"values": get_range(17, 34, 1e-4)},
        "noise_esp_decay": {"values": get_range(94, 100, 1e-2)},
        "shrink_ratio": {"values": get_range(51, 77, 1e-2)},
    },
}


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time)

    solver_param = DEFAULT_SOLVER_PARAM_M7_NORM
    solver_param.lr = wandb.config.lr
    solver_param.lr_weight_decay = wandb.config.lr_weight_decay
    solver_param.gamma = wandb.config.gamma
    solver_param.noise_esp = wandb.config.noise_esp
    solver_param.noise_esp_decay = wandb.config.noise_esp_decay
    solver_param.shrink_ratio = wandb.config.shrink_ratio
    solver_param.enable_load_model = ENABLE_LODE_MODEL

    trainer = Trainer(solver_param=solver_param)

    print(trainer.random_sample_solutions_with_evaluation(num_poses=100, num_sols=100))

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
