import numpy as np
import pandas as pd
from tabulate import tabulate

def display_ikp(l2, ang, avg_inference_time):
    print(
        tabulate(
            [[l2 * 1e3, np.rad2deg(ang), np.round(avg_inference_time * 1e3, decimals=0)]],
            headers=[
                "avg_l2 (mm)",
                "avg_ang (deg)",
                "avg_inference_time (ms)",
            ],
        )
    )