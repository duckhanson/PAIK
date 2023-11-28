# Import required packages
from datetime import datetime
import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint
import wandb
from paik.settings import SolverConfig
from paik.utils import init_seeds
from torch.utils.data import DataLoader
from paik.dataset import CustomDataset

from paik.solver import Solver, DEFAULT_SOLVER_PARAM_M7_NORM

USE_WANDB = False
PATIENCE = 4
POSE_ERR_THRESH = 6e-3


class Trainer(Solver):
    def __init__(self, solver_param: SolverConfig) -> None:
        super().__init__(solver_param)
        self.__noise_esp = self.param.noise_esp
        self.__noise_esp_decay = self.param.noise_esp_decay  # 0.8 - 0.9
        self.__std_scale = 1 / self.__noise_esp

    def mini_train(
        self,
        begin_time,
        num_epochs,
        batch_size=128,
        use_wandb=True,
        patience=4,
        pose_err_thres=1e-2,
        num_eval_poses=100,
        num_eval_sols=100,
    ) -> None:
        init_seeds()
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # data generation
        assert self._device == "cuda", "device should be cuda"

        update_noise_esp = lambda num_epochs: self.param.noise_esp * (
            self.__noise_esp_decay**num_epochs
        )

        train_loader = DataLoader(
            CustomDataset(
                features=self._J_tr, targets=np.column_stack((self._P_tr, self._F))
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            #   generator=torch.Generator(device='cuda')
        )

        self._solver.train()

        for ep in range(num_epochs):
            tqdm_train_loader = tqdm(train_loader)
            batch_loss = np.zeros((len(train_loader)))
            for i, batch in enumerate(tqdm_train_loader):
                loss = self.train_step(batch=batch)
                batch_loss[i] = loss
                bar = {"loss": f"{np.round(loss, 3)}"}
                tqdm_train_loader.set_postfix(bar, refresh=True)
                if np.isnan(loss):
                    print(f"Early stopping ({loss} is nan)")
                    break

            self.__noise_esp = update_noise_esp(ep)

            self.shrink_ratio = 0.25
            print(
                f"using shrink_ratio: {self.shrink_ratio} (fixed), where original shrink_ratio: {self.param.shrink_ratio} (training)"
            )
            avg_pos_errs, avg_ori_errs = self.random_evaluation(num_poses=num_eval_poses, num_sols=num_eval_sols)  # type: ignore
            self.shrink_ratio = self.param.shrink_ratio  # type: ignore

            self._scheduler.step(avg_pos_errs)  # type: ignore

            log_info = {
                "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
                "position_errors": avg_pos_errs,
                "orientation_errors": avg_ori_errs,
                "train_loss": batch_loss.mean(),
                "noise_esp": self.__noise_esp,
            }

            if use_wandb:
                wandb.log(log_info)
            else:
                pprint(log_info)

            if (
                np.isnan(avg_pos_errs)
                or avg_pos_errs > 1e-1
                or ep > 14
                and avg_pos_errs > 1.5e-2
            ):
                print(
                    f"Thresholds Touched ({avg_pos_errs} > 1e-1 or 1.5e-2 and ep > 14)"
                )
                break

            early_stopping(avg_pos_errs, self._solver)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if avg_pos_errs < pose_err_thres:  # type: ignore
                torch.save(
                    {
                        "solver": self._solver.state_dict(),
                        "opt": self._optimizer.state_dict(),
                    },
                    f"{self.param.weight_dir}/{begin_time}.pth",
                )

        print("Finished Training")

    def train_step(self, batch):
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
        x, y = add_noise(batch, esp=self.__noise_esp, std_scale=self.__std_scale)

        if self.param.enable_normalize:
            x, y = self.norm_J(x), self.norm_C(y)
        x = x.to(self._device)
        y = y.to(self._device)
        loss = -self._solver(y).log_prob(x)  # -log p(x | y)
        loss = loss.mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()


def add_noise(batch, esp: float, std_scale: float):
    J, C = batch
    if esp < 1e-9:
        std = torch.zeros((C.shape[0], 1))
        C = torch.column_stack((C, std))
    else:
        # softflow implementation
        s = esp * torch.rand((C.shape[0], 1))
        C = torch.column_stack((C, std_scale * s))
        noise = torch.normal(
            mean=torch.zeros_like(input=J),
            std=torch.repeat_interleave(input=s, repeats=J.shape[1], dim=1),
        )
        J = J + noise
    return J, C


class EarlyStopping:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        enable_save=False,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.enable_save = enable_save
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}, score: {score}, best: {self.best_score}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.enable_save:
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ..."
                )
            torch.save(model.state_dict(), self.path)
        else:
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f})."
                )
        self.val_loss_min = val_loss


def main() -> None:
    begin_time = datetime.now().strftime("%m%d-%H%M")
    # note that we define values from `wandb.config`
    # instead of defining hard values
    if USE_WANDB:
        wandb.init(name=begin_time, notes=f"r=0")

    solver_param = DEFAULT_SOLVER_PARAM_M7_NORM

    trainer = Trainer(solver_param=solver_param)

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

    if USE_WANDB:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()
    else:
        pprint(f"Finish job {begin_time}")


if __name__ == "__main__":
    init_seeds()
    main()
