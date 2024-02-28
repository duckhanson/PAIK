import numpy as np

def geometric_distance_between_quaternions(
        q1: np.ndarray, q2: np.ndarray
    ) -> np.ndarray:
    """
    Compute the geometric distance between two sets of quaternions. Reference from jrl.conversions.

    Args:
        q1 (np.ndarray): quaternions, shape: (num_quaternions, m - 3)
        q2 (np.ndarray): quaternions, shape: (num_quaternions, m - 3)

    Returns:
        np.ndarray: geometric distance, shape: (num_poses,)
    """
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

def evaluate_pose_error_P2d_P2d(P1: np.ndarray, P2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate pose error between two sets of poses

    Args:
        P1 (np.ndarray): poses, shape: (num_poses, m)
        P2 (np.ndarray): poses, shape: (num_poses, m)

    Returns:
        Tuple[np.ndarray, np.ndarray]: positional (l2, meters) and angular (ang, rads) errors
    """
    assert len(P1.shape) == 2 and len(P2.shape) == 2 and P1.shape[0] == P2.shape[0]
    l2 = np.linalg.norm(P1[:, :3] - P2[:, :3], axis=1)
    ang = geometric_distance_between_quaternions(
        P1[:, 3:], P2[:, 3:])
    return l2, ang

