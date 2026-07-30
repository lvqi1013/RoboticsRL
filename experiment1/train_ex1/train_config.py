from typing import Callable, NamedTuple
from dataclasses import asdict, dataclass
import numpy as np
import os
from pathlib import Path

CHEACKPOINT_OUTPUT_DIR = Path("results/checkpoint")
CHEACKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["subgoal_x", "subgoal_y"]
"""子目标的结果列名"""

BASE_FEATURE_COLS = ["start_x", "start_y", "goal_x", "goal_y", "yaw"]
"""训练的基础列名"""

class RegressionLabelStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray

@dataclass
class SubgoalResult:
    model: str
    seed: int
    mse: float
    rmse: float
    mae: float
    r2: float
    mde: float
    median_de: float
    in_bounds_rate: float
    valid_subgoal_rate: float
    obstacle_hit_rate: float
    reachable_segment_rate: float
    progress_rate: float
    mean_progress: float
    checkpoint: str    