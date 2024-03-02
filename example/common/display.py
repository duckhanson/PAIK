from __future__ import annotations
import numpy as np
import pandas as pd
from tabulate import tabulate


def display_ikp(l2: np.ndarray, ang: np.ndarray, avg_inference_time: float):
    """
    Display the average l2, ang, and inference time in a table format.

    Args:
        l2 (np.ndarray): mean of l2 distance between the generated IK solutions and the ground truth
        ang (np.ndarray): mean of quaternion distance between the generated IK solutions and the ground truth
        avg_inference_time (float): average inference time
    """
    print(
        tabulate(
            [
                [
                    l2 * 1e3,
                    np.rad2deg(ang),
                    np.round(avg_inference_time * 1e3, decimals=0),
                ]
            ],
            headers=[
                "avg_l2 (mm)",
                "avg_ang (deg)",
                "avg_inference_time (ms)",
            ],
        )
    )


def compute_success_rate(distance_J_deg: np.ndarray, success_distance_thresholds: list, return_percentage: bool=True) -> np.ndarray | list[str]:
    """
    Compute the success rate of the generated IK solutions for distance to desired joint configurations with respect to the success_distance_thresholds.

    Args:
        distance_J_deg (np.ndarray): distance between the generated IK solutions and the ground truth in degrees
        success_distance_thresholds (np.ndarray): success distance thresholds in degrees

    Returns:
        list[str]: success rate with respect to the success_distance_thresholds
    """
    success_rate = np.asarray([np.mean(distance_J_deg < th) * 100 for th in success_distance_thresholds])
        
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


def display_posture(record_dir, name, l2, ang, distance_J, success_distance_thresholds):
    """
    Display the posture of the generated IK solutions and save the posture to a pickle file.

    Args:
        record_dir (_type_): _description_
        name (_type_): _description_
        l2 (_type_): _description_
        ang (_type_): _description_
        distance_J (_type_): _description_
        success_distance_thresholds (_type_): _description_
    """
    
    assert l2.shape == ang.shape == distance_J.shape

    ang = np.rad2deg(ang)
    distance_J = np.rad2deg(distance_J)

    df = pd.DataFrame(
        {
            "l2 (m)": l2,
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
    
    success_rate_ikflow = compute_success_rate(df_ikflow["distance_J (deg)"].values, success_distance_thresholds, return_percentage=False)
    success_rate_nodeik = compute_success_rate(df_nodeik["distance_J (deg)"].values, success_distance_thresholds, return_percentage=False)
    success_rate_paik = compute_success_rate(df_paik["distance_J (deg)"].values, success_distance_thresholds, return_percentage=False)    
    
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
    ax = df.plot(x="Success Threshold (deg)", y=["IKFlow", "NodeIK", "PAIK"], grid=True, kind="line", title="Success Rate")
    ax.set_ylabel("Success Rate (%)", fontsize=fontsize)
    ax.set_xlabel("Distance Threshold (deg)", fontsize=fontsize)
    ax.set_title("Success Rate", fontsize=fontsize)
    ax.set_yticks(np.arange(0, 100, 15))
    ax.legend(fontsize=fontsize)