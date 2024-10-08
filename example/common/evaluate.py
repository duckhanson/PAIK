import numpy as np
import numpy.typing as npt
import torch
from typing import Any

def make_batches(arr: np.ndarray, batch_size: int):
    # Split the array into batches
    return np.split(arr, np.arange(batch_size, arr.shape[0], batch_size))


def batches_back_to_array(batches: list):
    # Concatenate the batches back into an array
    return np.concatenate(batches, axis=0)


def compute_distance_J(
    J_hat_3d: np.ndarray, J_ground_truth_2d: np.ndarray
) -> np.ndarray:
    """
    Compute the distance between the generated 3D IK solutions and the ground truth 2D IK solutions.

    Args:
        J_hat_3d (np.ndarray): IK solutions generated by the model with shape (num_sols, num_poses, num_dofs or n)
        J_ground_truth_2d (np.ndarray): IK solutions from the ground truth or numerical IK solutions with shape (num_poses, num_dofs or n)

    Returns:
        np.ndarray: distance between with shape (num_sols * num_poses)
    """

    assert (
        J_hat_3d.shape[1:] == J_ground_truth_2d.shape
    ), f"J_hat_3d.shape: {J_hat_3d.shape}, J_ground_truth_2d.shape: {J_ground_truth_2d.shape}, the last two dimensions of J_hat_3d and J_ground_truth_2d should be the same (num_poses, num_sols)."
    # J_hat_3d.shape = (num_sols, num_poses, num_dofs or n)
    # J_ground_truth_2d.shape = (num_poses, num_dofs or n)
    J_hat_2d = J_hat_3d.reshape(-1, J_hat_3d.shape[-1])
    J_ground_truth_2d = np.tile(J_ground_truth_2d, (J_hat_3d.shape[0], 1))

    # distance_J.shape = (num_sols * num_poses)
    return np.linalg.norm(J_hat_2d - J_ground_truth_2d, axis=-1).flatten()


def guassian_kernel(
    L2_distances_square: torch.Tensor, n_kernels: float = 5, mul_factor: float = 2.0
) -> torch.Tensor:
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
    bandwith = L2_distances_square.data.sum() / (n_samples**2 - n_samples)
    bandwidth_multipliers = mul_factor ** (
        torch.arange(n_kernels, device="cuda") - n_kernels // 2
    )

    kernels = torch.exp(
        -L2_distances_square[None, ...]
        / (bandwith * bandwidth_multipliers)[:, None, None]
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

    kernels = (
        widths**exponents
        * ((widths + L2_distances_square[None, ...]) / exponents) ** -exponents
    ).sum(dim=0)
    return kernels


def mmd_score(
    X: torch.Tensor,
    Y: torch.Tensor,
    use_inverse_multi_quadric: bool,
    mmd_kernels: torch.Tensor = torch.tensor([(0.2, 0.1), (0.2, 0.5), (0.2, 2)]),
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
    J_hat_: npt.NDArray[np.float32],
    J_ground_truth_: npt.NDArray[np.float32],
    num_poses: int,
    use_inverse_multi_quadric: bool = True,
) -> float:
    """
    _summary_

    Args:
        J_hat_ (npt.NDArray): generated IK solutions with shape (num_sols, num_poses, num_dofs or n)
        J_ground_truth_ (npt.NDArray): numerical IK solutions with shape (num_sols, num_poses, num_dofs or n)
        num_poses (int): the number of poses, which is the first dimension of J_hat_ and J_ground_truth_
        use_inverse_multi_quadric (bool, optional): if True, use inverse multiquadric kernel, else use gaussian kerenl. Defaults to True.

    Returns:
        float: a scalar of mmd score between the J_hat_ and J_ground_truth_ average over all poses
    """

    # check if J_hat.shape=(1, num_sols*num_poses, num_dofs)
    if J_hat_.shape[0] == 1 and J_ground_truth_.shape[0] == 1:
        J_hat_ = J_hat_.reshape(-1, num_poses, J_hat_.shape[-1])
        J_ground_truth_ = J_ground_truth_.reshape(-1, num_poses, J_ground_truth_.shape[-1])
    
    assert J_hat_.shape[1] == num_poses, f"J_hat_.shape[1]: {J_hat_.shape[1]}, num_poses: {num_poses}"
    assert J_ground_truth_.shape[1] == num_poses, f"J_ground_truth_.shape[1]: {J_ground_truth_.shape[1]}, num_poses: {num_poses}"

    device = "cuda"
    
    # if J_hat_ is a numpy array, convert it to a torch tensor
    if not torch.is_tensor(J_hat_):
        J_hat_ = torch.from_numpy(J_hat_).float().to(device)  # type: ignore
    
    if not torch.is_tensor(J_ground_truth_):
        J_ground_truth_ = (
            torch.from_numpy(J_ground_truth_).float().to(device)
        )  # type: ignore
        
    # cast J_hat_ and J_ground_truth_ to the same device
    J_hat_ = J_hat_.to(device)
    J_ground_truth_ = J_ground_truth_.to(device)
    
    mmd_all_poses = torch.stack(
        [
            mmd_score(J_hat_[:, i], J_ground_truth_[:, i], use_inverse_multi_quadric)
            for i in range(num_poses)
        ],
        dim=0,
    )
    mmd_mean = torch.mean(
        mmd_all_poses[~torch.isnan(mmd_all_poses)], dim=0, keepdim=True
    ).item()
    return round(mmd_mean, ndigits=2)


def geodesic_distance_between_quaternions(
    q1: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """
    Compute the geodesic distance between two sets of quaternions. Reference from jrl.conversions.

    Args:
        q1 (np.ndarray): quaternions, shape: (num_quaternions, m - 3)
        q2 (np.ndarray): quaternions, shape: (num_quaternions, m - 3)

    Returns:
        np.ndarray: geodesic distance, shape: (num_poses,)
    """
    if isinstance(q1, torch.Tensor):
        q1 = q1.detach().cpu().numpy()
    if isinstance(q2, torch.Tensor):
        q2 = q2.detach().cpu().numpy()
    
    acos_clamp_epsilon = 1e-7
    ang = np.abs(
        np.remainder(
            2
            * np.arccos(
                np.clip(
                    np.clip(np.sum(q1 * q2, axis=1), -1, 1),
                    -1 + acos_clamp_epsilon,
                    1 - acos_clamp_epsilon,
                )
            )
            + np.pi,
            2 * np.pi,
        )
        - np.pi
    )
    return ang


def evaluate_pose_error_P2d_P2d(
    P1: np.ndarray, P2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate pose error between two sets of poses

    Args:
        P1 (np.ndarray): poses, shape: (num_poses, m)
        P2 (np.ndarray): poses, shape: (num_poses, m)

    Returns:
        Tuple[np.ndarray, np.ndarray]: positional (l2, meters) and angular (ang, rads) errors
    """
    assert len(P1.shape) == 2 and len(
        P2.shape) == 2 and P1.shape[0] == P2.shape[0]
    l2 = np.linalg.norm(P1[:, :3] - P2[:, :3], axis=1)
    ang = geodesic_distance_between_quaternions(P1[:, 3:], P2[:, 3:])
    return l2, ang

def evaluate_pose_error_J3d_P2d(
    robot,
    J: np.ndarray,
    P: np.ndarray,
    return_posewise_evalution: bool = False,
    return_all: bool = False,
) -> tuple[Any, Any]:
    """
    Evaluate pose error given generated joint configurations J and ground truth poses P. Return default is l2 and ang with shape (1).

    Args:
        J (np.ndarray): generated joint configurations with shape (num_sols, num_poses, num_dofs)
        P (np.ndarray): ground truth poses with shape (num_poses, m)
        return_posewise_evalution (bool, optional): return l2 and ang with shape (num_poses,). Defaults to False.
        return_all (bool, optional): return l2 and ang with shape (num_sols * num_poses). Defaults to False.

    Returns:
        tuple[Any, Any]: l2 and ang, default shape (1), posewise evaluation with shape (num_poses,), or all evaluation with shape (num_sols * num_poses)
    """
    num_poses, num_sols = len(P), len(J)
    assert len(J.shape) == 3 and len(
        P.shape) == 2 and J.shape[1] == num_poses, f"J: {J.shape}, P: {P.shape}"

    # P: (num_poses, m), P_expand: (num_sols * num_poses, m)
    P_expand = np.tile(P, (num_sols, 1))
    
    # if J is torch.Tensor, cast it to np.ndarray
    if hasattr(J, "detach"):
        J = J.detach().cpu().numpy()
    
    P_hat = robot.forward_kinematics(J.reshape(-1, robot.n_dofs))
    l2, ang = evaluate_pose_error_P2d_P2d(P_hat, P_expand)  # type: ignore

    if return_posewise_evalution:
        return (
            l2.reshape(num_sols, num_poses).mean(axis=1),
            ang.reshape(num_sols, num_poses).mean(axis=1),
        )
    elif return_all:
        return l2, ang
    return l2.mean(), ang.mean()

def mmd_J3d_J3d(J1, J2, num_poses):
    """
    Calculate the maximum mean discrepancy between two sets of joint angles

    Args:
        J1 (np.ndarray): the first set of joint angles with shape (num_sols, num_poses, num_dofs or n)
        J2 (np.ndarray): the second set of joint angles with shape (num_sols, num_poses, num_dofs or n)
        
    Returns:
        float: the maximum mean discrepancy between the two sets of joint angles
    """
    # check num_poses is the same for J1 and J2
    if J1.shape[1] != num_poses or J2.shape[1] != num_poses:
        raise ValueError(f"The number of poses must be the same for both J1 ({J1.shape[1]}) and J2 ({J1.shape[1]})")
    mmd_score = mmd_evaluate_multiple_poses(J1, J2, num_poses)
    return mmd_score