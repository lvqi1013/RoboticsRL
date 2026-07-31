"""Geometry-aware metrics for predicted subgoals."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from experiment1.common_code.collision import is_position_valid, point_in_obstacle  # noqa: E402
from experiment1.common_code.data_structure import Point

MAP_BOUNDS = {
    4: {"x_min": -2.0, "x_max": 2.0, "y_min": -2.0, "y_max": 2.0},
    6: {"x_min": -3.0, "x_max": 3.0, "y_min": -3.0, "y_max": 3.0},
    10: {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0},
}


def in_bounds(point: np.ndarray, bounds: dict[str, float]) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        bounds["x_min"] <= x <= bounds["x_max"]
        and bounds["y_min"] <= y <= bounds["y_max"]
    )


def segment_is_free(
    start: np.ndarray,
    end: np.ndarray,
    bounds: dict[str, float],
    step_size: float = 0.05,
) -> bool:
    dist = float(np.linalg.norm(end - start))
    steps = max(1, int(math.ceil(dist / step_size)))
    for i in range(steps + 1):
        p = start + (end - start) * (i / steps)
        if not in_bounds(p, bounds):
            return False
        if point_in_obstacle(Point(float(p[0]), float(p[1])), bounds=bounds):
            return False
    return True


def subgoal_geometry_metrics(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    map_size: int,
) -> dict[str, float]:
    bounds = MAP_BOUNDS[map_size]
    starts = features[:, 0:2]
    goals = features[:, 2:4]

    error_vec = y_pred - y_true
    distance_errors = np.linalg.norm(error_vec, axis=1)
    goal_dist_before = np.linalg.norm(goals - starts, axis=1)
    goal_dist_after = np.linalg.norm(goals - y_pred, axis=1)

    in_bounds_flags = []
    valid_flags = []
    obstacle_flags = []
    segment_free_flags = []
    progress_flags = []

    for start, goal, pred, before, after in zip(
        starts, goals, y_pred, goal_dist_before, goal_dist_after
    ):
        pred_in_bounds = in_bounds(pred, bounds)
        point = Point(float(pred[0]), float(pred[1]))
        pred_obstacle = point_in_obstacle(point, bounds=bounds)
        pred_valid = is_position_valid(point, bounds=bounds)
        seg_free = segment_is_free(start, pred, bounds)
        progress = bool(after < before and np.linalg.norm(pred - start) > 0.2)

        in_bounds_flags.append(pred_in_bounds)
        obstacle_flags.append(pred_obstacle)
        valid_flags.append(pred_valid)
        segment_free_flags.append(seg_free)
        progress_flags.append(progress)

    return {
        "mde": float(np.mean(distance_errors)),
        "median_de": float(np.median(distance_errors)),
        "in_bounds_rate": float(np.mean(in_bounds_flags)),
        "valid_subgoal_rate": float(np.mean(valid_flags)),
        "obstacle_hit_rate": float(np.mean(obstacle_flags)),
        "reachable_segment_rate": float(np.mean(segment_free_flags)),
        "progress_rate": float(np.mean(progress_flags)),
        "mean_progress": float(np.mean(goal_dist_before - goal_dist_after)),
    }

