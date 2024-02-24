import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels, device="cuda") - n_kernels // 2
        )

        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        # assert L2_distances.device == "cuda" and self.bandwidth_multipliers.device == "cuda" and self.get_bandwidth(L2_distances).device == "cuda", f"L2_dis: {L2_distances.device}, bandmult: {self.bandwidth_multipliers.device}, band: {self.get_bandwidth(L2_distances).device}"
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        X_filtered = X[~torch.any(X.isnan(), dim=1)]
        Y_filtered = Y[~torch.any(Y.isnan(), dim=1)]

        if X_filtered.shape[0] == 0 or Y_filtered.shape[0] == 0:
            return torch.nan
        # print(f"X_filtered: {X_filtered.shape}, Y_filtered: {Y_filtered.shape}")
        K = self.kernel(torch.vstack([X_filtered, Y_filtered]))

        X_size = X_filtered.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def mmd_evaluate_multiple_poses(J_hat_, J_ground_truth_, num_poses):
    assert (
        len(J_hat_) == num_poses and len(J_ground_truth_) == num_poses
    ), f"J_hat_.shape: {J_hat_.shape}, num_poses: {num_poses}"
    mmd_score = MMDLoss()
    J_hat_ = torch.tensor(J_hat_, dtype=torch.float32, device="cuda")
    J_ground_truth_ = torch.tensor(J_ground_truth_, dtype=torch.float32, device="cuda")
    mmd_all_poses = torch.empty(num_poses, device="cuda")
    for i in range(num_poses):
        mmd_all_poses[i] = mmd_score(J_hat_[i], J_ground_truth_[i])
    mmd_all_poses = mmd_all_poses[~torch.isnan(mmd_all_poses)]
    mmd_all_poses = mmd_all_poses.cpu().numpy()
    return mmd_all_poses.mean()
