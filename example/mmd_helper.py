from typing import List, Tuple
import torch

def guassian_kernel(L2_distances, n_kernels=5, mul_factor=2.0):
    n_samples = L2_distances.shape[0]
    bandwith = L2_distances.data.sum() / (n_samples ** 2 - n_samples)
    bandwidth_multipliers = mul_factor ** (
        torch.arange(n_kernels, device="cuda") - n_kernels // 2
    )

    return torch.exp(
        -L2_distances[None, ...] / (bandwith * bandwidth_multipliers)[:, None, None]
    ).sum(dim=0)


def inverse_multiquadric_kernel(
    L2_distances, widths_exponents: List[Tuple[float, float]]
):
    kernel_val = [
        width ** exponent * ((width + L2_distances) / exponent) ** -exponent
        for width, exponent in widths_exponents
    ]
    return sum(kernel_val)


def mmd_score(
    X, Y, use_inverse_multi_quadric, mmd_kernels=[(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
):
    X = X[~torch.any(X.isnan(), dim=1)]
    Y = Y[~torch.any(Y.isnan(), dim=1)]

    min_len = min(X.shape[0], Y.shape[0])
    X, Y = X[:min_len], Y[:min_len]

    if min_len == 0:
        return torch.nan

    total = torch.vstack([X, Y])
    L2_distances = torch.cdist(total, total).square()

    if use_inverse_multi_quadric:
        kernels = inverse_multiquadric_kernel(L2_distances, mmd_kernels)
    else:
        kernels = guassian_kernel(L2_distances)

    XX = kernels[:min_len, :min_len]
    YY = kernels[min_len:, min_len:]
    XY = kernels[:min_len, min_len:]
    return (XX - 2 * XY + YY).mean()


def mmd_evaluate_multiple_poses(
    J_hat_, J_ground_truth_, num_poses, use_inverse_multi_quadric=True
):
    assert (
        len(J_hat_) == num_poses and len(J_ground_truth_) == num_poses
    ), f"J_hat_.shape: {J_hat_.shape}, num_poses: {num_poses}"

    J_hat_ = torch.from_numpy(J_hat_).float().to("cuda")
    J_ground_truth_ = torch.from_numpy(J_ground_truth_).float().to("cuda")
    mmd_all_poses = torch.tensor(
        [
            mmd_score(J_hat_[i], J_ground_truth_[i], use_inverse_multi_quadric)
            for i in range(num_poses)
        ],
        device="cuda",
    )
    mmd_mean = mmd_all_poses[~torch.isnan(mmd_all_poses)].mean()
    return mmd_mean.item()