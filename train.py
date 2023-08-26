# Import required packages
import torch
from tqdm import tqdm

import wandb
# from utils.dataset import create_dataset
from utils.model import *
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import *

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


def mini_train(config=None,
               robot=None,
               J_tr=None,
               P_tr=None,
               P_ts=None,
               F=None,
               knn=None) -> None:
    # data generation
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_loader = get_train_loader(J=J_tr,
                                    P=P_tr,
                                    F=F,
                                    device=device,
                                    batch_size=config["batch_size"])

    # Build Generative model, NSF
    # Neural spline flow (NSF) with 3 sample features and 5 context features
    solver, optimizer, scheduler = get_flow_model(
        load_model=cfg.use_pretrained,
        num_transforms=config["num_transforms"],
        subnet_width=config["subnet_width"],
        subnet_num_layers=config["subnet_num_layers"],
        lr=config["lr"],
        lr_weight_decay=config["lr_weight_decay"],
        decay_step_size=config["decay_step_size"],
        gamma=config["gamma"],
        device=device)

    solver.train()

    for ep in range(config['num_epochs']):
        t = tqdm(train_loader)
        step = 0
        batch_loss = np.zeros((len(train_loader)))
        for batch in t:
            loss = train_step(model=solver,
                              batch=batch,
                              optimizer=optimizer,
                              scheduler=scheduler)
            batch_loss[step] = loss
            bar = {"loss": f"{np.round(loss, 3)}"}
            t.set_postfix(bar, refresh=True)

            step += 1

        rand = np.random.randint(low=0, high=len(P_ts), size=cfg.num_eval_size)
        position_errors, orientation_errors, _ = test(
            robot=robot,
            P_ts=P_ts[rand],
            F=F,
            solver=solver,
            knn=knn,
            K=cfg.K,
            print_report=False,
        )
        wandb.log({
            'position_errors': position_errors.mean(),
            'train_loss': batch_loss.mean(),
        })

        if ep % 3 == 1:
            torch.save(solver.state_dict(), cfg.path_solver)
    print("Finished Training")


def main() -> None:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="msik_train",
        # entity
        entity="luca_nthu",
        # track hyperparameters and run metadata
        config=cfg,
        notes=f'r={cfg.r}',
    )
    robot = Robot(verbose=False)
    J_tr, P_tr = data_collection(robot=robot, N=cfg.N_train)
    _, P_ts = data_collection(robot=robot, N=cfg.N_test)
    F = posture_feature_extraction(J_tr)
    knn = get_knn(P_tr=P_tr)

    config = {
        'subnet_width': cfg.subnet_width,
        'subnet_num_layers': cfg.subnet_num_layers,
        'num_transforms': cfg.num_transforms,
        'lr': cfg.lr,
        'lr_weight_decay': cfg.lr_weight_decay,
        'decay_step_size': cfg.decay_step_size,
        'gamma': cfg.decay_gamma,
        'batch_size': cfg.batch_size,
        'num_epochs': cfg.num_epochs,
    }

    mini_train(config=config,
               robot=robot,
               J_tr=J_tr,
               P_tr=P_tr,
               P_ts=P_ts,
               F=F,
               knn=knn)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


if __name__ == "__main__":
    main()
