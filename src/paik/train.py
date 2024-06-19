# Import required packages
from datetime import datetime
import torch
import torch.backends.cudnn
import numpy as np

from tqdm import tqdm
from pprint import pprint
import wandb
from torch.utils.data import DataLoader, TensorDataset
from .settings import SolverConfig, PANDA_PAIK
from .solver import Solver

PATIENCE = 4
POSE_ERR_THRESH = 6e-3


def init_seeds(seed=42):
    print(f"set seed {seed}")
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed
    )  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed
    )  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer(Solver):
    def __init__(self, solver_param: SolverConfig) -> None:
        super().__init__(solver_param)
        self.__noise_esp = self.param.noise_esp
        self.__noise_esp_decay = self.param.noise_esp_decay  # 0.8 - 0.9
        self.__noise_scale = 1 / self.__noise_esp

    def mini_train(
        self,
        begin_time,
        num_epochs,
        batch_size=128,
        patience=4,
        pose_err_thres=1e-2,
        num_eval_poses=100,
        num_eval_sols=100,
        seed=42,
    ) -> None:
        init_seeds(seed=seed)
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            enable_save=True,
            path=f"{self.param.weight_dir}/{begin_time}.pth",
            val_loss_threshold=pose_err_thres,
        )
        # data generation
        assert self._device == "cuda", "device should be cuda"

        def update_noise_esp(num_epochs):
            return self.param.noise_esp * (self.__noise_esp_decay**num_epochs)

        self._solver.train()

        for ep in range(num_epochs):
            # add noise
            if self.__noise_esp < 1e-9:
                noise_std = np.zeros((len(self.F), 1))
            else:
                noise_std = self.__noise_esp * np.random.rand(len(self.F), 1)
            J = self.normalize_input_data(
                self.J + noise_std * np.random.randn(*self.J.shape), "J"
            )
            C = self.normalize_input_data(
                np.column_stack(
                    (self.P, self.F, self.__noise_scale * noise_std)),
                "C",
            )
            C = self.remove_posture_feature(C) if self._use_nsf_only else C

            train_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(J.astype(np.float32)).to(self._device),
                    torch.from_numpy(C.astype(np.float32)).to(self._device),
                ),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                # generator=torch.Generator(device='cuda:0'), # when use train stand alone
            )

            tqdm_train_loader = tqdm(train_loader)
            batch_loss = np.zeros((len(train_loader)))
            for i, batch in enumerate(tqdm_train_loader):
                tqdm_train_loader.set_description(f"Epochs {ep}")

                loss = self.train_step(batch=batch)
                batch_loss[i] = loss
                bar = {"loss": f"{np.round(loss, 3)}"}
                tqdm_train_loader.set_postfix(bar, refresh=True)
                if np.isnan(loss):
                    print(f"Early stopping ({loss} is nan)")
                    break

            self.__noise_esp = update_noise_esp(ep)

            avg_pos_errs, avg_ori_errs, _, _ = self.evaluate_ikp_iterative(  # type: ignore
                num_poses=num_eval_poses, num_sols=num_eval_sols, verbose=False
            )  # type: ignore
            self.base_std = self.param.base_std  # type: ignore

            self._scheduler.step(avg_pos_errs)  # type: ignore

            log_info = {
                "lr": self._optimizer.param_groups[0]["lr"],  # type: ignore
                "position_errors": avg_pos_errs,
                "orientation_errors": avg_ori_errs,
                "train_loss": batch_loss.mean(),
                "noise_esp": self.__noise_esp,
            }

            wandb.log(log_info)

            early_stopping(avg_pos_errs, self._solver,
                           self._optimizer)  # type: ignore

            if early_stopping.early_stop:
                print("Early stopping")
                break

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
        x, y = batch
        loss = -self._solver(y).log_prob(x)  # -log p(x | y)
        loss = loss.mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()


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
        val_loss_threshold=0.0,
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
        self.early_stop = False
        self.val_loss_threshold = val_loss_threshold
        self.val_loss_min = np.Inf
        self.delta = delta
        self.enable_save = enable_save
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer):
        if np.isnan(val_loss):
            self.early_stop = True
            self.trace_func(f"EarlyStopping counter: val_loss: nan")
        elif self.val_loss_min is None or val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, model, optimizer=optimizer)
            self.counter = 0
        elif val_loss - self.val_loss_min > self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}, val_loss: {val_loss}, best: {self.val_loss_min}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer):
        """Saves model when validation loss decrease."""
        if self.enable_save:
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f})."
                )

            if val_loss < self.val_loss_threshold:
                self.trace_func(f"Saving model to {self.path}.")
                torch.save(
                    {
                        "solver": model.state_dict(),
                        "opt": optimizer.state_dict(),
                    },
                    self.path,
                )
        else:
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f})."
                )
        self.val_loss_min = val_loss
