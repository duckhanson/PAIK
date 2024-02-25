from typing import List, Tuple
import numpy as np
import numpy.typing as npt
import torch

def guassian_kernel(L2_distances_square: torch.Tensor, n_kernels:float=5, mul_factor: float=2.0) -> torch.Tensor:
    """
    Apply the gaussian kernel to the L2 distances of two distributions. Multiple kernels are used to capture different scales of the data.

    Args:
        L2_distances_square (torch.Tensor): computed by torch.cdist(X, Y).square()
        n_kernels (float, optional): numbers of gaussian kernels. Defaults to 5.
        mul_factor (float, optional): the base bandwidth of a gaussian kernel. Defaults to 2.0.

    Returns:
        torch.Tensor: distance matrix of the two distributions after applying the gaussian kernel
    """
    n_samples = L2_distances_square.shape[0]
    bandwith = L2_distances_square.data.sum() / (n_samples ** 2 - n_samples)
    bandwidth_multipliers = mul_factor ** (
        torch.arange(n_kernels, device="cuda") - n_kernels // 2
    )

    kernels = torch.exp(
        -L2_distances_square[None, ...] / (bandwith * bandwidth_multipliers)[:, None, None]
    ).sum(dim=0)
    
    return kernels


def inverse_multiquadric_kernel(
    L2_distances_square: torch.Tensor, widths_exponents: torch.Tensor
) -> torch.Tensor:
    """
    Apply the inverse multiquadric kernel to the L2 distances of two distributions.

    Args:
        L2_distances (torch.Tensor): computed by torch.cdist(X, Y).square()
        widths_exponents (torch.Tensor): backward mmd as defined in https://github.com/vislearn/analyzing_inverse_problems

    Returns:
        torch.Tensor: distance matrix of the two distributions after applying the inverse multiquadric kernel
    """
    widths_exponents = widths_exponents.to(L2_distances_square.device)
    widths, exponents = torch.unbind(widths_exponents, dim=1)
    widths = widths[:, None, None]
    exponents = exponents[:, None, None]
    
    kernels = (widths ** exponents * ((widths + L2_distances_square[None, ...]) / exponents) ** -exponents).sum(dim=0)
    return kernels


def mmd_score(
    X: torch.Tensor, Y: torch.Tensor, use_inverse_multi_quadric: bool, mmd_kernels: torch.Tensor=torch.tensor([(0.2, 0.1), (0.2, 0.5), (0.2, 2)])
) -> torch.Tensor:
    """
    compute the maximum mean discrepancy (MMD) between two distributions X and Y. Two types of kernels are used: gaussian and inverse multiquadric and only the latter will use mmd_kernels.

    Args:
        X (torch.Tensor): one distribution, in our case, the generated IK solutions
        Y (torch.Tensor): one distribution, in our case, the ground truth IK solutions or the numerical IK solutions
        use_inverse_multi_quadric (bool): use inverse multiquadric kernel or not
        mmd_kernels (torch.Tensor, optional): the base bandwidth and exponents of an inverse multiquadric kernel. Defaults to torch.tensor([(0.2, 0.1), (0.2, 0.5), (0.2, 2)]) as defined in https://github.com/vislearn/analyzing_inverse_problems.

    Returns:
        torch.Tensor: a scalar of mmd score between the X and Y
    """

    X = X[~X.isnan().any(dim=1)]
    Y = Y[~Y.isnan().any(dim=1)]

    min_len = min(X.shape[0], Y.shape[0])

    if min_len == 0:
        return torch.tensor(float("nan"), device=X.device)
    
    X, Y = X[:min_len], Y[:min_len]

    total = torch.vstack([X, Y])
    L2_distances_square = torch.cdist(total, total).square()

    if use_inverse_multi_quadric:
        kernels = inverse_multiquadric_kernel(L2_distances_square, mmd_kernels)
    else:
        kernels = guassian_kernel(L2_distances_square)

    XX = kernels[:min_len, :min_len]
    YY = kernels[min_len:, min_len:]
    XY = kernels[:min_len, min_len:]
    return (XX - 2 * XY + YY).mean()


def mmd_evaluate_multiple_poses(
    J_hat_: npt.NDArray[np.float32], J_ground_truth_: npt.NDArray[np.float32], num_poses: int, use_inverse_multi_quadric: bool=True
) -> float:
    """
    _summary_

    Args:
        J_hat_ (npt.NDArray): generated IK solutions with shape (num_poses, num_sols, num_dofs or n)
        J_ground_truth_ (npt.NDArray): numerical IK solutions with shape (num_poses, num_sols, num_dofs or n)
        num_poses (int): the number of poses, which is the first dimension of J_hat_ and J_ground_truth_
        use_inverse_multi_quadric (bool, optional): if True, use inverse multiquadric kernel, else use gaussian kerenl. Defaults to True.

    Returns:
        float: a scalar of mmd score between the J_hat_ and J_ground_truth_ average over all poses
    """
    
    assert (
        len(J_hat_) == num_poses and J_hat_.shape == J_ground_truth_.shape
    ), f"J_hat_.shape: {J_hat_.shape}, J_ground_truth_.shape: {J_ground_truth_.shape}, num_poses: {num_poses}"
    
    device = "cuda"
    J_hat_ = torch.from_numpy(J_hat_).float().to(device) # type: ignore
    J_ground_truth_ = torch.from_numpy(J_ground_truth_).float().to(device) # type: ignore
    mmd_all_poses = torch.stack(
        [
            mmd_score(J_hat_[i], J_ground_truth_[i], use_inverse_multi_quadric)
            for i in range(num_poses)
        ],
        dim=0,
    )
    mmd_mean = torch.mean(mmd_all_poses[~torch.isnan(mmd_all_poses)], dim=0, keepdim=True)
    return mmd_mean.item()