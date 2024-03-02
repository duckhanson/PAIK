import numpy as np
import pandas as pd

def save_diversity(record_dir: str, name: str, J_hat: np.ndarray, l2: np.ndarray, ang: np.ndarray, mmd: np.ndarray, base_std: list):
    assert J_hat.shape[0] == l2.shape[0] == ang.shape[0] == mmd.shape[0] == len(base_std)
    
    np.save(f"{record_dir}/{name}_J_hat.npy", J_hat)
    
    df = pd.DataFrame({
        "l2": l2,
        "ang": np.rad2deg(ang),
        "mmd": mmd,
        "base_std": base_std,
    })
    
    df.to_pickle(f"{record_dir}/{name}_std.pkl")
    
    print(f"[SUCCESS] saved to {record_dir}/{name}_std.pkl")
    
    
    