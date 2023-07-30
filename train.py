# Import required packages
import time
import os

import flaml
# import numpy as np
# import pandas as pd
import torch
from ray import tune
# import zuko
# from hnne import HNNE
# from torch import Tensor, nn
from tqdm import tqdm

import wandb
# from utils.dataset import create_dataset
from utils.model import *
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import *

# from zuko.distributions import BoxUniform, DiagNormal, Minimum
# from zuko.flows import NSF, Distribution, DistributionModule, FlowModule, Unconditional


def train_step(model, batch, optimizer, scheduler):
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
    x, y = add_noise(batch)

    loss = -model(y).log_prob(x)  # -log p(x | y)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


def mini_train(config, checkpoint_dir, robot):
    # sweep_config = {
    #     "subnet_width": tune.randint(128, 2048),   # log transformed with base 2
    #     "subnet_num_layers": tune.randint(3, 20),   # log transformed with base 2
    #     "num_transforms": tune.randint(3, 25)
    #     "lr": tune.loguniform(1e-8, 1e-4),
    #     "num_epochs": tune.loguniform(1, max_num_epoch),
    #     "batch_size": tune.randint(128, 1024)    # log transformed with base 2
    # }
    # data generation
    J_tr, P_tr = data_collection(robot=robot, N=cfg.N_train)
    J_ts, P_ts = data_collection(robot=robot, N=cfg.N_test)
    F = posture_feature_extraction(J_tr)
    train_loader = get_train_loader(J=J_tr, P=P_tr, F=F, batch_size=config["batch_size"])
    # Build Generative model, NSF
    # Neural spline flow (NSF) with 3 sample features and 5 context features
    solver, optimizer, scheduler = get_flow_model(
        load_model=cfg.use_pretrained,
        num_transforms=config["num_transforms"],
        subnet_width=config["subnet_width"],
        subnet_num_layers=config["subnet_num_layers"],
        lr=config["lr"])
    knn = load_pickle(file_path=cfg.path_knn)
    
    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        solver.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    solver.train()

    for ep in range(config["num_epochs"]):
        t = tqdm(train_loader)
        step = 0
        for batch in t:
            loss = train_step(
                model=solver, batch=batch, optimizer=optimizer, scheduler=scheduler
            )

            bar = {"loss": f"{np.round(loss, 3)}"}
            t.set_postfix(bar, refresh=True)

            step += 1

            if step % cfg.num_steps_eval == 0:
                rand = np.random.randint(low=0, high=len(P_ts), size=cfg.num_eval_size)
                test(
                    robot=panda,
                    P_ts=P_ts[rand],
                    F=F,
                    solver=solver,
                    knn=knn,
                    K=cfg.K,
                    print_report=True,
                )

            if step % cfg.num_steps_save == 0:
                torch.save(solver.state_dict(), cfg.path_solver)
        
        rand = np.random.randint(low=0, high=len(P_ts), size=cfg.num_eval_size)
        _, position_errors, _ = test(
            robot=panda,
            P_ts=P_ts[rand],
            F=F,
            solver=solver,
            knn=knn,
            K=cfg.K,
            print_report=False,
        )
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=ep) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (solver.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=position_errors)
    print("Finished Training")


if __name__ == "__main__":
    panda = Robot(verbose=False)

    max_num_epoch = 5
    gpus_per_trial = 0.5    # number of gpus for each trial; 0.5 means two training jobs can share one gpu
    time_budget_s = 600     # time budget in seconds
    num_samples = 500       # maximal number of trials
    np.random.seed(7654321)
    
    sweep_config = {
        "subnet_width": tune.randint(128, 2048),   # log transformed with base 2
        "subnet_num_layers": tune.randint(3, 20),   # log transformed with base 2
        "num_transforms": tune.randint(3, 25),
        "lr": tune.loguniform(1e-8, 1e-4),
        "num_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(128, 1024)    # log transformed with base 2
    }

    start_time = time.time()
    result = flaml.tune.run(
        tune.with_parameters(mini_train, robot=panda),
        config=sweep_config,
        metric="loss",
        mode="min",
        low_cost_partial_config={"num_epochs": 1},
        max_resource=max_num_epoch,
        min_resource=1,
        scheduler="asha",  # Use asha scheduler to perform early stopping based on intermediate results reported
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        local_dir='logs/',
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        use_ray=True)


    print(f"#trials={len(result.trials)}")
    print(f"time={time.time()-start_time}")
    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metric_analysis["loss"]["min"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.metric_analysis["accuracy"]["max"]))

    best_trained_model = Net(2**best_trial.config["l1"],
                            2**best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_value = getattr(best_trial.checkpoint, "dir_or_data", None) or best_trial.checkpoint.value
    checkpoint_path = os.path.join(checkpoint_value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_acc = _test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="ikpflow",
    #     # entity
    #     entity="luca_nthu",
    #     # track hyperparameters and run metadata
    #     config=cfg,
    # )

    # step = 0
    # l2_label = ["mean", "std", "min", "25%", "50%", "75%", "max"]

    # flow.train()
    # for ep in range(cfg.num_epochs):
    #     t = tqdm(train_loader)
    #     for batch in t:
    #         loss = train_step(
    #             model=flow, batch=batch, optimizer=optimizer, scheduler=scheduler
    #         )

    #         bar = {"loss": f"{np.round(loss, 3)}"}
    #         t.set_postfix(bar, refresh=True)

    #         # log metrics to wandb
    #         wandb.log({"loss": np.round(loss, 3)})

    #         step += 1

    #         if step % cfg.num_steps_eval == 0:
    #             df = test(
    #                 robot=panda,
    #                 P_ts=P_ts[: cfg.num_eval_size],
    #                 F=F,
    #                 solver=solver,
    #                 knn=knn,
    #                 K=100,
    #                 print_report=True,
    #             )
    #             l2_val = df.describe().values[1:, 0]
    #             log_info = {}
    #             for l, v in zip(l2_label, l2_val):
    #                 log_info[l] = v
    #             log_info["learning_rate"] = scheduler.get_last_lr()[0]
    #             wandb.log(log_info)

    #         if step % cfg.num_steps_save == 0:
    #             torch.save(flow.state_dict(), cfg.path_solver)

    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()
