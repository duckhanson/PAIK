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


def display_posture(
    record_dir: str,
    name: str,
    l2: np.ndarray,
    ang: np.ndarray,
    distance_J: np.ndarray,
    success_distance_thresholds: list,
):
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


def display_posture_all(
    iksolver_names: list, record_dir: str, success_distance_thresholds: list
):
    iksolver_dfs = {}
    try:
        iksolver_dfs = {
            name: pd.read_pickle(f"{record_dir}/{name.lower()}_posture.pkl")
            for name in iksolver_names
        }
    except:
        print("Please run display_posture for every IK solver first.")
        return
    iksolver_success_rates = {
        name: compute_success_rate(
            df["distance_J (deg)"].values,
            success_distance_thresholds,
            return_percentage=False,
        )
        for name, df in iksolver_dfs.items()
    }

    # plot success rate with respect to success_distance_thresholds
    df = pd.DataFrame(
        {
            "Success Threshold (deg)": success_distance_thresholds,
            **{
                name: success_rates
                for name, success_rates in iksolver_success_rates.items()
            },
        }
    )
    print(df.describe())

    fontsize = 28
    title_fontsize = fontsize + 4
    figsize = (9, 8)
    ax = df.plot(
        x="Success Threshold (deg)",
        y=[name for name in iksolver_names],
        grid=True,
        kind="line",
        title="Success Rate",
        fontsize=fontsize,
        figsize=figsize,
    )
    ax.set_ylabel("Success Rate (%)", fontsize=title_fontsize)
    ax.set_xlabel("Distance Threshold (deg)", fontsize=title_fontsize)
    ax.set_title("Success Rate", fontsize=title_fontsize)
    ax.set_yticks(np.arange(0, 100, 15))
    ax.legend(fontsize=title_fontsize)
    plt.show()


def display_diversity_all(iksolver_names: list, record_dir: str):
    try:
        iksolver_dfs = {
            name: pd.read_pickle(f"{record_dir}/{name.lower()}_std.pkl")
            for name in iksolver_names
        }
    except:
        print("Please run diversity.py for every IK solver first. For instance, run:")
        print("python example/diversity.py")
        print("conda activate nodeik")
        print("python nodeik_experiments.py: diversity() function.")
        return

    df_l2 = pd.DataFrame(
        {
            "Base std": iksolver_dfs[iksolver_names[0]].base_std.values,
            **{name: df.l2.values * 1000 for name, df in iksolver_dfs.items()},
        }
    )
    
    df_mmd = pd.DataFrame(
        {
            "Base std": iksolver_dfs[iksolver_names[0]].base_std.values,
            **{name: df.mmd.values for name, df in iksolver_dfs.items()},
        }
    )

    # plot diversity with respect to base_stds
    fontsize = 24
    title_fontsize = fontsize + 4
    figsize = (14, 7)
    ax_figsize = (1, 1)
    
    fig, sub_ax = plt.subplots(1,2, figsize=figsize)
    x_ticks = np.arange(0, 1.5, 0.3)[1:]
    df_l2.plot(x="Base std", grid=True, fontsize=fontsize, ax=sub_ax[0], legend=False)
    sub_ax[0].set_xticks(x_ticks)
    sub_ax[0].set_xlabel("Base std", fontsize=title_fontsize)
    sub_ax[0].set_ylabel("L2 Error (mm)", fontsize=title_fontsize)
    sub_ax[0].set_title("Position Error", fontsize=title_fontsize)
    # sub_ax[0].figure.set_size_inches(ax_figsize)
    # sub_ax[0].legend(fontsize=fontsize)

    df_mmd.plot(x="Base std", grid=True, fontsize=fontsize, ax=sub_ax[1], legend=False)
    sub_ax[1].set_xticks(x_ticks)
    sub_ax[1].set_xlabel("Base std", fontsize=title_fontsize)
    sub_ax[1].set_ylabel("MMD Score", fontsize=title_fontsize)
    sub_ax[1].set_title("MMD Score", fontsize=title_fontsize)
    # sub_ax[1].figure.set_size_inches(ax_figsize)
    # sub_ax[1].legend(fontsize=fontsize)
    fig.legend(iksolver_names, loc="center right", bbox_to_anchor=(0.9, 0.7), fontsize=title_fontsize)
    
    plt.subplots_adjust(wspace=0.3)
    plt.show()
