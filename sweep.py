# Import required packages
from datetime import datetime

from tqdm import tqdm

import wandb
from utils.model import *
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import *

early_stop_param = {
    'train_loss': -1,
    'position_error': 4.8e-2,
}

sweep_config = {
    'name': 'sweep',
    'method': 'random',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            'values': [900, 1024, 1200, 1400]
            # 'value': 1024
        },
        'subnet_num_layers': {
            'values': [3, 4]
            # 'value': 3
        },
        'num_transforms': {
            'values': [7, 8, 11, 14, 16]  # 6, 8, ..., 16
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 4e-4,
            'max': 6.5e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 5e-3,
            'max': 5e-2,
        },
        'decay_step_size': {
            # 'values': [3e4, 4e4],
            'value': 4e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-2,
            'min': 4e-1,
            'max': 7e-1,
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 3
        }
    },
}


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
               knn=None,
               begin_time=None) -> None:
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
        _, position_errors, orientation_errors, _ = test(
            robot=robot,
            P_ts=P_ts[rand],
            F=F,
            solver=solver,
            knn=knn,
            K=cfg.K,
            print_report=False,
        )
        wandb.log({
            'epoch': ep,
            'position_errors': position_errors.mean(),
            'orientation_errors': orientation_errors.mean(),
            'train_loss': batch_loss.mean(),
        })

        if ep == 0 and position_errors.mean(
        ) > early_stop_param['position_error']:
            break
    model_weights_path =  cfg.weight_dir + begin_time + '.pth'
    torch.save({
        'solver': solver.state_dict(),
        'opt': optimizer.state_dict(),
        }, model_weights_path)
    del train_loader
    del scheduler
    del optimizer
    del solver
    print("Finished Training")


def main() -> None:
    robot = Robot(verbose=False)
    J_tr, P_tr = data_collection(robot=robot, N=cfg.N_train)
    _, P_ts = data_collection(robot=robot, N=cfg.N_test)
    F = posture_feature_extraction(J=J_tr)
    knn = get_knn(P_tr=P_tr)
    
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    wandb.init(name=begin_time,
                     notes=f'm={cfg.m}')

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
               robot=robot,
               J_tr=J_tr,
               P_tr=P_tr,
               P_ts=P_ts,
               F=F,
               knn=knn,
               begin_time=begin_time)


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project='msik_sweep',
                           entity='luca_nthu')
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=30)
