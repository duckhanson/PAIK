import numpy as np
import pandas as pd
from tabulate import tabulate

def display_ikp(l2, ang, avg_inference_time):
    print(
        tabulate(
            [[l2, np.rad2deg(ang), avg_inference_time]],
            headers=[
                "avg_l2 (m)",
                "avg_ang (deg)",
                "avg_inference_time (s)",
            ],
        )
    )