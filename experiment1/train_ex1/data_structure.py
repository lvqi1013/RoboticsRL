"""保存实验中所要用到的数据结构"""
from dataclasses import dataclass
@dataclass
class Point:
    """
    二维坐标点
    """
    x: float
    y: float

@dataclass(frozen=True)
class GridPoint:
    """
    离散化连续坐标点的网格坐标点
    """
    gx: int
    gy: int

@dataclass(frozen=True)  # 使用 frozen=True 确保配置对象在创建后不可变，防止在数据采集过程中被意外修改
class GenerateDatasetConfig:
    """
    数据集生成的全局配置参数。
    用于记录数据集的生成条件，并通常会随数据集一起保存为 metadata.json 文件，
    以确保实验的完全可复现性。
    """

    map_size: int
    """仿真地图的尺寸等级（如 4, 6, 10），对应不同复杂度的迷宫环境"""

    world_name: str
    # Gazebo 仿真世界名称（例如 "maze"），用于环境加载和校验

    samples: int
    """计划采集的有效样本总数"""

    lookahead: float
    """随机种子，用于控制环境重置、起点/终点生成及采样的可复现性"""

    astar_resolution: float
    """A* 路径规划时的空间分辨率（单位：米），影响路径的平滑度和计算耗时"""

    min_distance: float
    """起点与全局目标之间的最小物理距离，用于过滤掉过于简单的样本"""

    max_wait_for_observation: float
    """每次环境重置后，等待传感器数据（位姿、Lidar等）稳定的最大超时时间（秒）"""

    output: str 
    """生成的 CSV 数据集文件的保存路径"""