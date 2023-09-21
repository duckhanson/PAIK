# Import required packages
from datetime import datetime
import torch 
import numpy as np

from tqdm import tqdm
from pprint import pprint
import wandb
from utils.model import get_knn, get_flow_model
from utils.robot import Robot
from utils.settings import config as cfg
from utils.utils import load_all_data, EarlyStopping, get_train_loader, train_step, test


USE_WANDB = False
# NUM_RECORD_STEPS = 14e3
PATIENCE = 4    

def mini_train(config=None,
               begin_time=None) -> None:
    
    robot = Robot(verbose=False)
    J_tr, P_tr, P_ts, F = load_all_data(robot)
    knn = get_knn(P_tr=P_tr)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, delta=1e-4)
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
        enable_load_model=cfg.use_pretrained,
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
        batch_loss = np.zeros((len(train_loader)))
        step = 0
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
        avg_position_error, avg_orientation_error, _ = test(
            robot=robot,
            P_ts=P_ts[rand],
            F=F,
            solver=solver,
            knn=knn,
            K=cfg.K,
            print_report=False,
        )
        
        if USE_WANDB:
            wandb.log({
                'ep': ep,
                'position_errors': avg_position_error,
                'orientation_errors': avg_orientation_error,
                'train_loss': batch_loss.mean(),
            })
        else:
            pprint({
                'ep': ep,
                'position_errors': avg_position_error,
                'orientation_errors': avg_orientation_error,
                'train_loss': batch_loss.mean(),
            })

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_position_error, solver)
                
        if early_stopping.early_stop:
            print("Early stopping")
            break   
                
    model_weights_path =  cfg.weight_dir + begin_time + '.pth'
    
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
    if USE_WANDB:   
        wandb.init(name=begin_time,
                         notes=f'r=0')
    
    # note that we define values from `wandb.config`
    # instead of defining hard values
    config = {
        'subnet_width': 1024,
        'subnet_num_layers': 3,
        'num_transforms': 12,
        'lr': 5e-4,
        'lr_weight_decay': 2.7e-2,
        'decay_step_size': 4e4,
        'gamma': 9.79e-2,
        'batch_size': 128,
        'num_epochs': 10,
    }

    mini_train(config=config,
               begin_time=begin_time)

    if USE_WANDB:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
    else:
        pprint(f"Finish job {begin_time}")


if __name__ == "__main__":
    main()
