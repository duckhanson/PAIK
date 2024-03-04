from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate


def display_ikp(l2: np.ndarray, ang: np.ndarray, inference_time: float):
    """
    Display the average l2, ang, and inference time in a table format.

    Args:
        l2 (np.ndarray): mean of l2 distance between the generated IK solutions and the ground truth
        ang (np.ndarray): mean of quaternion distance between the generated IK solutions and the ground truth
        inference_time (float): average inference time
    """
    print(
        tabulate(
            [
                [
                    l2 * 1e3,
                    np.rad2deg(ang),
                    np.round(inference_time * 1e3, decimals=0),
                ]
            ],
            headers=[
                "l2 (mm)",
                "ang (deg)",
                "inference_time (ms)",
            ],
        )
    )


def compute_success_rate(
    distance_J_deg: np.ndarray,
    success_distance_thresholds: list,
    return_percentage: bool = True,
) -> np.ndarray | list[str]:
    """
    Compute the success rate of the generated IK solutions for distance to desired joint configurations with respect to the success_distance_thresholds.

    Args:
        distance_J_deg (np.ndarray): distance between the generated IK solutions and the ground truth in degrees
        success_distance_thresholds (np.ndarray): success distance thresholds in degrees

    Returns:
        list[str]: success rate with respect to the success_distance_thresholds
    """
    success_rate = np.asarray(
        [np.mean(distance_J_deg < th) * 100 for th in success_distance_thresholds]
    )

    if return_percentage:
        return [np.round(rate, decimals=1) for rate in success_rate]
    return success_rate


def display_success_rate(distance_J_deg: np.ndarray, success_distance_thresholds: list):
    """
    Display the success rate of the generated IK solutions for distance to desired joint configurations with respect to the success_distance_thresholds.

    Args:
        distance_J_deg (np.ndarray): distance between the generated IK solutions and the ground truth in degrees
        success_distance_thresholds (list): success distance thresholds in degrees
    """
    print(
        tabulate(
            {
                "Success Threshold (deg)": success_distance_thresholds,
                "Success Rate": compute_success_rate(
                    distance_J_deg, success_distance_thresholds
                ),
            },
            headers="keys",
            tablefmt="pretty",
        )
    )


def display_posture(record_dir: str, name: str, l2: np.ndarray, ang: np.ndarray, distance_J: np.ndarray, success_distance_thresholds: list):
    """
    Display the posture of the generated IK solutions and save the posture to a pickle file.

    Args:
        record_dir (str): the path to the record directory, e.g. "{workdir}/record/{date}"
        name (str): the name of the IK solver, e.g. "ikflow", "nodeik", "paik"
        l2 (np.ndarray): position error with unit in meters (m)
        ang (np.ndarray): orientation (quaternion) error with unit in radians (rads)
        distance_J (np.ndarray): distance between the generated IK solutions and the ground truth in radians (rads)
        success_distance_thresholds (list): success distance thresholds in degrees
    """

    assert l2.shape == ang.shape == distance_J.shape

    ang = np.rad2deg(ang)
    distance_J = np.rad2deg(distance_J)

    df = pd.DataFrame(
        {
            "l2 (mm)": l2 * 1e3,
            "ang (deg)": ang,
            "distance_J (deg)": distance_J,
        }
    )
    print(df.describe())
    df.to_pickle(f"{record_dir}/{name}_posture.pkl")
    display_success_rate(distance_J, success_distance_thresholds)


def display_posture_all(record_dir, success_distance_thresholds):
    try:
        df_ikflow = pd.read_pickle(f"{record_dir}/ikflow_posture.pkl")
        df_nodeik = pd.read_pickle(f"{record_dir}/nodeik_posture.pkl")
        df_paik = pd.read_pickle(f"{record_dir}/paik_posture.pkl")
    except:
        print("Please run display_posture for every IK solver first.")
        return

    success_rate_ikflow = compute_success_rate(
        df_ikflow["distance_J (deg)"].values,
        success_distance_thresholds,
        return_percentage=False,
    )
    success_rate_nodeik = compute_success_rate(
        df_nodeik["distance_J (deg)"].values,
        success_distance_thresholds,
        return_percentage=False,
    )
    success_rate_paik = compute_success_rate(
        df_paik["distance_J (deg)"].values,
        success_distance_thresholds,
        return_percentage=False,
    )

    # plot success rate with respect to success_distance_thresholds
    df = pd.DataFrame(
        {
            "Success Threshold (deg)": success_distance_thresholds,
            "IKFlow": success_rate_ikflow,
            "NodeIK": success_rate_nodeik,
            "PAIK": success_rate_paik,
        }
    )

    print(df.describe())

    fontsize = 24
    figsize = (9, 8)
    ax = df.plot(
        x="Success Threshold (deg)",
        y=["IKFlow", "NodeIK", "PAIK"],
        grid=True,
        kind="line",
        title="Success Rate",
        fontsize=fontsize,
        figsize=figsize,
    )
    ax.set_ylabel("Success Rate (%)", fontsize=fontsize)
    ax.set_xlabel("Distance Threshold (deg)", fontsize=fontsize)
    ax.set_title("Success Rate", fontsize=fontsize)
    ax.set_yticks(np.arange(0, 100, 15))
    ax.legend(fontsize=fontsize)
    plt.show()


def display_diversity_all(record_dir: str):
    try:
        df_paik = pd.read_pickle(f"{record_dir}/paik_posture_mmd_std.pkl")
        df_ikflow = pd.read_pickle(f"{record_dir}/ikflow_posture_mmd_std.pkl")
        df_nodeik = pd.read_pickle(f"{record_dir}/nodeik_posture_mmd_std.pkl")
    except:
        print("Please run diversity.py for every IK solver first.")
        return

    # plot diversity with respect to base_stds
    fontsize = 24
    figsize = (9, 8)
    df_l2 = pd.DataFrame(
        {
            "PAIK": df_paik.l2.values * 1000,
            "IKFlow": df_ikflow.l2.values * 1000,
            "NODEIK": df_nodeik.l2.values * 1000,
            "Base std": df_paik.base_std.values,
        }
    )

    ax = df_l2.plot(x="Base std", grid=True, fontsize=fontsize, figsize=figsize)
    ax.set_xlabel("Base std", fontsize=fontsize)
    ax.set_ylabel("L2 Error (mm)", fontsize=fontsize)
    ax.set_title("Position Error", fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    df_mmd = pd.DataFrame(
        {
            "PAIK": df_paik.mmd.values,
            "IKFlow": df_ikflow.mmd.values,
            "NODEIK": df_nodeik.mmd.values,
            "Base std": df_paik.base_std.values,
        }
    )

    ax1 = df_mmd.plot(x="Base std", grid=True, fontsize=fontsize, figsize=figsize)
    ax1.set_xlabel("Base std", fontsize=fontsize)
    ax1.set_ylabel("MMD Score", fontsize=fontsize)
    ax1.set_title("MMD Score", fontsize=fontsize)
    ax1.legend(fontsize=fontsize)
    plt.show()
