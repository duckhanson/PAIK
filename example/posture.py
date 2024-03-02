from common.display import display_posture_all
from common.config import ConfigPosture

if __name__ == "__main__":
    config = ConfigPosture()
    config.date = "2024_03_02"

    display_posture_all(config.record_dir, config.success_distance_thresholds)
