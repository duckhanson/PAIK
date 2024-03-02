import numpy as np
import pandas as pd
from tabulate import tabulate


def display_ikp(l2, ang, avg_inference_time):
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


def compute_success_rate(distance_J_deg, success_distance_thresholds):
    success_rate = []
    for th in success_distance_thresholds:
        success_rate.append(
            f"{np.round(np.mean(distance_J_deg < th) * 100, decimals=1)}%"
        )
    return success_rate


def display_success_rate(distance_J_deg, success_distance_thresholds):
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
