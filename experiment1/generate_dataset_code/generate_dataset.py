"""Generate A*-supervised subgoal datasets without launching Gazebo."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC = REPO_ROOT / "src" / "turtlebot4_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from geometry_metrics import MAP_BOUNDS  # noqa: E402
from experiment1.common_code.astar_planner import Astar  # noqa: E402
from experiment1.common_code.collision import is_position_valid, point_in_obstacle  # noqa: E402
from experiment1.common_code.data_structure import Point  # noqa: E402


MIN_DISTANCE = {4: 2.0, 6: 4.0, 10: 7.0}


@dataclass(frozen=True)
class _AstarEnv:
    map_bounds: dict[str, float]


def sample_point(
    rng: random.Random,
    bounds: dict[str, float],
    clearance: float,
) -> tuple[float, float]:
    for _ in range(10000):
        x = round(rng.uniform(bounds["x_min"], bounds["x_max"]), 3)
        y = round(rng.uniform(bounds["y_min"], bounds["y_max"]), 3)
        if is_position_valid(Point(x, y), bounds=bounds, clearance=clearance):
            return (x, y)
    raise RuntimeError("Could not sample a valid point.")


def outside_bounds(x: float, y: float, bounds: dict[str, float]) -> bool:
    return x < bounds["x_min"] or x > bounds["x_max"] or y < bounds["y_min"] or y > bounds["y_max"]


def raycast_lidar(
    position: tuple[float, float],
    yaw: float,
    bounds: dict[str, float],
    beams: int,
    max_range: float,
    step: float,
) -> list[float]:
    readings: list[float] = []
    base_x, base_y = position
    for beam_idx in range(beams):
        angle = yaw - math.pi + (2.0 * math.pi * beam_idx / beams)
        distance = max_range
        steps = int(max_range / step)
        for s in range(1, steps + 1):
            d = s * step
            x = base_x + d * math.cos(angle)
            y = base_y + d * math.sin(angle)
            if outside_bounds(x, y, bounds) or point_in_obstacle(Point(x, y), bounds=bounds):
                distance = d
                break
        readings.append(round(distance, 4))
    return readings


def path_length(path: list[tuple[float, float]]) -> float:
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def lookahead_subgoal(
    path: list[tuple[float, float]],
    lookahead: float,
) -> tuple[float, float]:
    if len(path) < 2:
        return path[-1]

    remaining = lookahead
    for a, b in zip(path, path[1:]):
        ax, ay = a
        bx, by = b
        seg_len = math.hypot(bx - ax, by - ay)
        if seg_len <= 1e-8:
            continue
        if remaining <= seg_len:
            ratio = remaining / seg_len
            return (ax + (bx - ax) * ratio, ay + (by - ay) * ratio)
        remaining -= seg_len
    return path[-1]


def generate_dataset(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    bounds = MAP_BOUNDS[args.map_size]
    env = _AstarEnv(bounds)

    # 创建 Astar 实例
    astar_planner = Astar(resolution=args.astar_resolution, env=env)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "yaw",
        "subgoal_x",
        "subgoal_y",
    ] + [f"lidar_{i}" for i in range(args.lidar_beams)]

    accepted = 0
    attempts = 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        while accepted < args.samples and attempts < args.max_attempts:
            attempts += 1
            start = sample_point(rng, bounds, args.clearance)
            goal = sample_point(rng, bounds, args.clearance)
            if math.hypot(goal[0] - start[0], goal[1] - start[1]) < MIN_DISTANCE[args.map_size]:
                continue

            # 使用 Astar 实例的 run_astar 方法
            path_points = astar_planner.run_astar(Point(start[0], start[1]), Point(goal[0], goal[1]))
            if not path_points or len(path_points) < 2:
                continue
            # 将 Point 对象列表转换为元组列表
            path = [(p.x, p.y) for p in path_points]
            if path_length(path) < args.min_path_length:
                continue

            yaw = rng.uniform(-math.pi, math.pi)
            subgoal = lookahead_subgoal(path, args.lookahead)
            lidar = raycast_lidar(
                start,
                yaw,
                bounds,
                beams=args.lidar_beams,
                max_range=args.lidar_max_range,
                step=args.lidar_step,
            )
            writer.writerow(
                [
                    f"{start[0]:.4f}",
                    f"{start[1]:.4f}",
                    f"{goal[0]:.4f}",
                    f"{goal[1]:.4f}",
                    f"{yaw:.4f}",
                    f"{subgoal[0]:.4f}",
                    f"{subgoal[1]:.4f}",
                    *[f"{v:.4f}" for v in lidar],
                ]
            )
            accepted += 1
            if accepted % args.report_every == 0:
                print(f"accepted={accepted} attempts={attempts}")

    if accepted != args.samples:
        raise RuntimeError(f"Generated {accepted}/{args.samples} samples.")
    print(f"Wrote {accepted} samples to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-size", type=int, choices=[4, 6, 10], required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clearance", type=float, default=0.51)
    parser.add_argument("--astar-resolution", type=float, default=0.10)
    parser.add_argument("--min-path-length", type=float, default=0.4)
    parser.add_argument("--lidar-beams", type=int, default=64)
    parser.add_argument("--lidar-max-range", type=float, default=15.0)
    parser.add_argument("--lidar-step", type=float, default=0.05)
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--report-every", type=int, default=1000)
    return parser


def main() -> int:
    return generate_dataset(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
