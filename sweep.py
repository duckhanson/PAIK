# Import required packages
from datetime import datetime

from tqdm import tqdm

import wandb
from utils.model import *
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import *

NUM_RECORD_STEPS = 2e4

sweep_config = {
    'name': 'sweep',
    'method': 'random',
    'metric': {
        'name': 'position_errors',
        'goal': 'minimize'
    },
    'parameters': {
        'subnet_width': {
            'values': [1024, 1250, 1450, 1700]
            # 'value': 1600
        },
        'subnet_num_layers': {
            # 'values': [3, 4]
            'value': 3
        },
        'num_transforms': {
            'values': [8, 13, 15]  # 6, 8, ..., 16
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-5,
            'min': 3e-4,
            'max': 4.5e-4,
        },
        'lr_weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'q_uniform',
            'q': 1e-3,
            'min': 1.6e-2,
            'max': 3.5e-2,
        },
        'decay_step_size': {
            'values': [5e4, 6e4, 8e4, 1e5],
            # 'value': 4e4
        },
        'gamma': {
            'distribution': 'q_uniform',
            'q': 1e-2,
            'min': 5e-2,
            'max': 3e-1,
        },
        'batch_size': {
            'value': 128
        },
        'num_epochs': {
            'value': 2
        }
    },
}
    

def mini_train(config=None,
               begin_time=None) -> None:
    
    robot = Robot(verbose=False)
    J_tr, P_tr, P_ts, F = load_all_data(robot)
    knn = get_knn(P_tr=P_tr)
    early_stopping = EarlyStopping(patience=3, verbose=True, delta=1e-4)
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
            
            if step % NUM_RECORD_STEPS == NUM_RECORD_STEPS-1:
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
                    'step': step,
                    'position_errors': position_errors.mean(),
                    'orientation_errors': orientation_errors.mean(),
                    'train_loss': batch_loss[:step].mean(),
                })

                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(position_errors.mean(), solver)
                
                if early_stopping.early_stop:
                    break
        if early_stopping.early_stop:
                print("Early stopping")
                break   
                
    model_weights_path =  cfg.weight_dir + begin_time + '.pth'
    
    if position_errors.mean() < 6e-3:
        torch.save({
            'solver': solver.state_dict(),
            'opt': optimizer.state_dict(),
            }, model_weights_path)
    del robot
    del J_tr
    del P_tr
    del P_ts
    del F
    del knn
    del train_loader
    del scheduler
    del optimizer
    del solver
    print("Finished Training")


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
               begin_time=begin_time)


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project='msik_sweep_25M_r4',
                           entity='luca_nthu')
    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=10)
