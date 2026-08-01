"""Generate TabM subgoal data from a running TurtleBot4 Gazebo simulation.

Start Gazebo first:

    ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py model:=lite world:=maze

Then run this script from the turtlebot4_lite_drl repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nav_rule_based import TurtleBotNavEnv
from experiment1.common_code.astar_planner import Astar
from experiment1.common_code.collision import is_position_valid
from experiment1.common_code.data_structure import Point
from geometry_metrics import MAP_BOUNDS, segment_is_free
from generate_dataset import lookahead_subgoal, path_length  


MIN_DISTANCE = {4: 2.0, 6: 4.0, 10: 7.0}
DEFAULT_LOOKAHEAD = {4: 1.5, 6: 2.0, 10: 2.0}

@dataclass(frozen=True)  # 使用 frozen=True 确保配置对象在创建后不可变，防止在数据采集过程中被意外修改
class DatasetConfig:
    """
    数据集生成的全局配置参数。
    用于记录数据集的生成条件，并通常会随数据集一起保存为 metadata.json 文件，
    以确保实验的完全可复现性。
    """
    
    map_size: int             # 仿真地图的尺寸等级（如 4, 6, 10），对应不同复杂度的迷宫环境
    world_name: str           # Gazebo 仿真世界名称（例如 "maze"），用于环境加载和校验
    samples: int              # 计划采集的有效样本总数
    seed: int                 # 随机种子，用于控制环境重置、起点/终点生成及采样的可复现性
    lookahead: float          # A* 前瞻距离（单位：米），用于从全局路径中截取局部子目标（Subgoal）
    astar_resolution: float   # A* 路径规划时的空间分辨率（单位：米），影响路径的平滑度和计算耗时
    min_distance: float       # 起点与全局目标之间的最小物理距离，用于过滤掉过于简单的样本
    max_wait_for_observation: float  # 每次环境重置后，等待传感器数据（位姿、Lidar等）稳定的最大超时时间（秒）
    output: str               # 生成的 CSV 数据集文件的保存路径
    launch_command: str       # 启动 Gazebo 仿真环境所需的完整 ROS2 Launch 命令，记录在元数据中以便复现

def csv_header() -> list[str]:
    return [
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "yaw",
        "subgoal_x",
        "subgoal_y",
        *[f"lidar_{i}" for i in range(64)],
    ]

def append_header_if_needed(path: Path) -> None:
    """
    确保目标 CSV 文件存在，并且如果它是一个新文件（或空文件），就自动为其写入表头（Header）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_header())

def count_existing_samples(path: Path) -> int:
    """
    统计目标 CSV 文件中已经成功采集到的有效样本数量
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)

def finite_xy(value: object) -> np.ndarray | None:
    """安全地提取并校验二维坐标 (x, y)。"""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr[:2].copy()

def current_global_goal(env: TurtleBotNavEnv) -> np.ndarray | None:
    """从仿真环境对象中安全、兼容地提取当前的全局目标（Global Goal）坐标。"""
    goal = getattr(env, "global_goal_position", None)
    if goal is None:
        goal = getattr(env, "goal_position", None)
    return finite_xy(goal)


def wait_for_stable_sample(env: TurtleBotNavEnv, timeout: float) -> bool:
    """
    等待仿真环境中的传感器数据初始化完成并达到稳定状态。
    在 Gazebo 中，刚 reset 时物理引擎和传感器插件可能需要几帧时间来准备数据，
    此函数通过轮询确保采集到的第一帧数据是合法且可用的。
    
    Args:
        env: TurtleBot4 导航仿真环境对象。
        timeout: 最大等待超时时间（秒），超时则放弃本次采集。
        
    Returns:
        bool: 如果在超时前获取到稳定数据返回 True，否则返回 False。
    """
    start = time.time()  # 记录开始等待的时间戳
    
    # 在超时时间内持续轮询检查
    while time.time() - start < timeout:
        # 触发 ROS2 节点的时钟更新，确保仿真时间与系统时间同步
        env.node.get_clock().now()
        
        # 处理一次 ROS2 消息回调，使传感器话题（如 /scan, /odom）的数据得以更新
        import rclpy
        rclpy.spin_once(env.node, timeout_sec=0.05)
        
        # 1. 安全提取并校验当前机器人的二维坐标 (x, y)
        position = finite_xy(getattr(env, "current_position", None))
        
        # 2. 获取 64 束激光雷达数据
        lidar = getattr(env, "lidar_data_64", None)
        
        # 3. 获取当前机器人的偏航角 (Yaw)
        yaw = float(getattr(env, "current_yaw", 0.0))
        
        # 综合校验所有关键传感器数据是否均已就绪且合法
        if (
            position is not None          # 坐标不为空（finite_xy 已校验过 NaN/Inf）
            and lidar is not None         # 激光雷达数据不为空
            and np.asarray(lidar).shape[0] == 64  # 激光雷达通道数必须严格为 64
            and np.all(np.isfinite(lidar))        # 所有激光雷达读数必须是有限数值（无 NaN/Inf）
            and math.isfinite(yaw)                # 偏航角必须是有限数值
        ):
            return True  # 所有条件满足，数据已稳定，返回成功
            
    return False  # 超过最大等待时间，数据仍未稳定，返回失败

def build_label(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    env: TurtleBotNavEnv,
    lookahead: float,
    astar_resolution: float,
) -> tuple[tuple[float, float] | None, list[tuple[float, float]] | None, str | None]:
    """
    根据当前起点和终点，计算 A* 全局路径并提取前瞻子目标标签。
    
    Args:
        start: 当前机器人二维坐标 (x, y)。
        goal: 全局目标点二维坐标 (x, y)。
        env: 仿真环境对象，用于提供地图边界和障碍物信息。
        lookahead: 前瞻距离（米），决定子目标在路径上距离当前位置多远。
        astar_resolution: A* 算法的栅格分辨率（米/像素）。
        
    Returns:
        tuple: (subgoal, path, error_reason)
            - subgoal: 合法的前瞻子目标坐标，失败时为 None。
            - path: 完整的 A* 路径点列表，A* 失败时为 None。
            - error_reason: 错误原因字符串，成功时为 None。
    """
    # 1. 创建 A* 实例并调用 run_astar 方法计算从 start 到 goal 的全局无碰撞路径
    astar_planner = Astar(resolution=astar_resolution, env=env)
    path_points = astar_planner.run_astar(
        Point(float(start[0]), float(start[1])),
        Point(float(goal[0]), float(goal[1]))
    )
    
    # 2. 校验 A* 是否成功找到有效路径（至少需要起点+终点 2 个点）
    if not path_points or len(path_points) < 2:
        return None, None, "astar_failed"
    
    # 将 Point 对象列表转换为元组列表
    path = [(p.x, p.y) for p in path_points]
    
    # 3. 过滤过短路径：如果起终点距离小于 0.3m，认为已到达或无需导航
    if path_length(path) < 0.3:
        return None, path, "path_too_short"

    # 4. 沿路径按 lookahead 距离提取前瞻子目标点
    subgoal = lookahead_subgoal(path, lookahead)
    
    # 5. 校验子目标是否在地图合法边界内
    if not is_position_valid(Point(float(subgoal[0]), float(subgoal[1])), bounds=env.map_bounds):
        return None, path, "subgoal_invalid"
    
    # 6. 校验从当前位置到子目标的直线段是否被障碍物阻挡
    #    （防止 A* 路径绕障但前瞻距离过大导致子目标穿墙）
    if not segment_is_free(start, np.asarray(subgoal, dtype=np.float32), env.map_bounds):
        return None, path, "segment_blocked"
    
    # 7. 所有校验通过，返回合法的子目标、完整路径和无错误标识
    return subgoal, path, None

def write_metadata(config: DatasetConfig, output: Path) -> None:
    """
    将数据集的配置信息和格式规范序列化为 JSON 元数据文件。
    该文件与数据文件同名但后缀为 .metadata.json，用于记录数据集的生成上下文，
    确保后续训练或分析时能够准确理解数据的结构、来源及采集参数。

    Args:
        config: 数据集采集配置对象，包含所有影响数据生成的超参数。
        output: 数据输出文件的路径（如 data.csv），元数据文件将基于此路径生成。
    """
    # 构建元数据字典，包含四个核心维度的信息
    metadata = {
        # 1. 完整记录采集时的配置参数，保证实验可复现
        "config": asdict(config),
        
        # 2. 动态获取 CSV 表头列表，作为程序读取时的字段索引依据
        "columns": csv_header(),
        
        # 3. 人类可读的格式模板，直观展示数据列的物理含义与排列顺序
        "format": "start_x,start_y,goal_x,goal_y,yaw,subgoal_x,subgoal_y,lidar_0,...,lidar_63",
        
        # 4. 数据来源声明，明确标注数据来自真实 Gazebo 仿真而非合成/噪声数据
        "note": "Samples use real Gazebo pose/yaw/LaserScan and A* lookahead labels.",
    }
    
    # 将输出路径的后缀替换为 .metadata.json，生成配套的元数据文件名
    meta_path = output.with_suffix(".metadata.json")
    
    # 序列化并写入文件
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",  # 格式化 JSON，保留中文，末尾加换行符
        encoding="utf-8",  # 显式指定 UTF-8 编码，避免跨平台乱码
    )

def make_env(args: argparse.Namespace) -> TurtleBotNavEnv:
    """
    根据命令行参数创建并返回配置好的 TurtleBot4 导航仿真环境实例。
    充当 CLI 参数与环境构造函数之间的适配层，隔离外部输入与内部初始化逻辑。

    Args:
        args: 由 argparse 解析得到的命令行参数命名空间，
              包含 map_size、max_wait_for_observation 等运行时配置。

    Returns:
        TurtleBotNavEnv: 初始化完成的仿真环境对象，可直接用于数据采集或训练。
    """
    # 根据命令行指定的地图尺寸名称，从全局常量表中查找对应的物理边界 (x_min, x_max, y_min, y_max)
    bounds = MAP_BOUNDS[args.map_size]

    return TurtleBotNavEnv(
        # === 以下参数来自命令行，支持运行时灵活调整 ===
        max_wait_for_observation=args.max_wait_for_observation,  # 等待传感器数据就绪的最大超时时间(秒)
        map_bounds=bounds,                                       # 当前地图的有效导航区域边界
        min_distance=args.min_distance,                          # 起点与终点之间的最小生成距离(m)，避免无效短路径样本
        positions_file=args.positions_file,                      # 预设的合法起终点坐标文件路径(可选)

        # === 以下为数据集采集专用的硬编码参数 ===
        subgoal_mode="none",          # 禁用环境内置的子目标计算，因为子目标标签由外部 build_label() 独立生成
        goal_reach_threshold=0.2,     # 判定到达全局目标的距离阈值(m)
        subgoal_reach_threshold=0.2,  # 判定到达子目标的距离阈值(m)，此处保留仅为满足构造函数签名要求
    )    


def generate(args: argparse.Namespace) -> int:
    if args.world_name != "maze":
        raise ValueError(
            "The current TurtleBot environment code uses /world/maze services. "
            "Launch with world:=maze or update the environment before using another world name."
        )
    seed = args.seed
    map_size = args.map_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.lookahead is None:
        args.lookahead = DEFAULT_LOOKAHEAD[args.map_size]
    if args.min_distance is None:
        args.min_distance = MIN_DISTANCE[args.map_size]

    launch_command = (
        "ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py "
        "model:=lite world:=maze"
    )
    args.output = args.output / f"subgoal_gazebo_maze_map{map_size}_seed{seed}.csv"

    config = DatasetConfig(
        map_size=args.map_size,
        world_name=args.world_name,
        samples=args.samples,
        seed=args.seed,
        lookahead=args.lookahead,
        astar_resolution=args.astar_resolution,
        min_distance=args.min_distance,
        max_wait_for_observation=args.max_wait_for_observation,
        output=str(args.output),
        launch_command=launch_command,
    )

    append_header_if_needed(args.output)
    write_metadata(config, args.output)

    accepted = count_existing_samples(args.output) if args.resume else 0
    attempts = 0
    rejected: dict[str, int] = {}
    print(f"Expected running Gazebo command: {launch_command}")
    print(f"Writing to {args.output}; existing accepted samples={accepted}")

    env = make_env(args)
    try:
        with args.output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            while accepted < args.samples and attempts < args.max_attempts:
                attempts += 1
                try:
                    env.reset(seed=args.seed + attempts)
                except Exception as exc:
                    key = f"reset_error:{type(exc).__name__}"
                    rejected[key] = rejected.get(key, 0) + 1
                    print(f"[WARN] reset failed at attempt={attempts}: {exc}")
                    continue

                if not getattr(env, "reset_success", True):
                    rejected["reset_not_success"] = rejected.get("reset_not_success", 0) + 1
                    continue
                if not wait_for_stable_sample(env, args.max_wait_for_observation):
                    rejected["observation_timeout"] = rejected.get("observation_timeout", 0) + 1
                    continue

                start = finite_xy(getattr(env, "current_position", None))
                goal = current_global_goal(env)
                lidar = np.asarray(getattr(env, "lidar_data_64", None), dtype=np.float32)
                yaw = float(getattr(env, "current_yaw", 0.0))
                if start is None or goal is None or lidar.shape[0] != 64 or not math.isfinite(yaw):
                    rejected["bad_state"] = rejected.get("bad_state", 0) + 1
                    continue

                subgoal, _path, reason = build_label(
                    start=start,
                    goal=goal,
                    env=env,
                    lookahead=args.lookahead,
                    astar_resolution=args.astar_resolution,
                )
                if subgoal is None:
                    rejected[reason or "label_failed"] = rejected.get(reason or "label_failed", 0) + 1
                    continue

                writer.writerow(
                    [
                        f"{start[0]:.4f}",
                        f"{start[1]:.4f}",
                        f"{goal[0]:.4f}",
                        f"{goal[1]:.4f}",
                        f"{yaw:.4f}",
                        f"{subgoal[0]:.4f}",
                        f"{subgoal[1]:.4f}",
                        *[f"{float(v):.4f}" for v in lidar],
                    ]
                )
                f.flush()
                accepted += 1

                if accepted % args.report_every == 0:
                    print(
                        f"accepted={accepted}/{args.samples} attempts={attempts} "
                        f"rejected={rejected}"
                    )

        if accepted < args.samples:
            raise RuntimeError(
                f"Only collected {accepted}/{args.samples} samples after {attempts} attempts. "
                f"Rejected summary: {rejected}"
            )
        print(f"Wrote {accepted} samples to {args.output}")
        return 0
    finally:
        try:
            env.close()
        except Exception:
            pass
        
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-size", type=int, choices=[4, 6, 10], default=10)
    parser.add_argument("--world-name", default="maze")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead", type=float, default=None)
    parser.add_argument("--min-distance", type=float, default=None)
    parser.add_argument("--astar-resolution", type=float, default=0.10)
    parser.add_argument("--max-wait-for-observation", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=300000)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--positions-file", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"experiment1/results/dataset_from_gazebo/"),
    )
    return parser


def main() -> int:
    return generate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())