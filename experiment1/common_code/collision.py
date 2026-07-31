"""Static obstacle geometry shared by spawning, A*, and subgoal validation."""
from __future__ import annotations
from .data_structure import Point
from typing import Iterable


Rectangle = tuple[float, float, float, float]


OBSTACLES_BY_MAP: dict[int, list[Rectangle]] = {
    4: [
        (-2.005 - 0.01 / 2, -2.0, 0.01, 4.0),
        (2.005 - 0.01 / 2, -2.0, 0.01, 4.0),
        (-2.0, -2.005 - 0.01 / 2, 4.0, 0.01),
        (-2.0, 2.005 - 0.01 / 2, 4.0, 0.01),
        (-1.05 - 0.2 / 2, -1.2 - 0.1 / 2, 0.2, 0.1),
        (1.05 - 0.3 / 2, 1.25 - 0.1 / 2, 0.3, 0.1),
        (-0.5 - 0.3 / 2, 1.1 - 0.2 / 2, 0.3, 0.2),
        (0.1 - 0.2 / 2, -0.2 - 0.2 / 2, 0.2, 0.2),
        (1.1 - 0.2 / 2, -1.05 - 0.2 / 2, 0.2, 0.2),
        (-1.7 - 0.3 / 2, -0.15 - 0.3 / 2, 0.3, 0.3),
    ],
    6: [
        (-3.005 - 0.01 / 2, -3.0, 0.01, 6.0),
        (3.005 - 0.01 / 2, -3.0, 0.01, 6.0),
        (-3.0, -3.005 - 0.01 / 2, 6.0, 0.01),
        (-3.0, 3.005 - 0.01 / 2, 6.0, 0.01),
        (2.05 - 0.4 / 2, 1.85 - 0.1 / 2, 0.4, 0.1),
        (-1.05 - 0.3 / 2, -2.1 - 0.8 / 2, 0.3, 0.8),
        (-0.35 - 0.2 / 2, -0.7 - 0.2 / 2, 0.2, 0.2),
        (1.1 - 0.2 / 2, -2.05 - 0.2 / 2, 0.2, 0.2),
        (-2.7 - 0.3 / 2, -0.15 - 0.3 / 2, 0.3, 0.3),
        (0.8 - 0.1 / 2, 0.8 - 0.1 / 2, 0.1, 0.1),
        (1.35 - 0.3 / 2, -0.45 - 0.1 / 2, 0.3, 0.1),
        (-2.25 - 0.5 / 2, 1.35 - 0.3 / 2, 0.5, 0.3),
        (-0.9 - 0.4 / 2, 2.25 - 0.2 / 2, 0.4, 0.2),
    ],
    10: [
        (-5.005 - 0.01 / 2, -5.0, 0.01, 10.0),
        (5.005 - 0.01 / 2, -5.0, 0.01, 10.0),
        (-5.0, -5.005 - 0.01 / 2, 10.0, 0.01),
        (-5.0, 5.005 - 0.01 / 2, 10.0, 0.01),
        (-4.125, 1.3, 0.25, 2.4),
        (1.5, 2.0, 2.0, 1.0),
        (3.0, 1.0, 1.0, 1.0),
        (-1.90, -2.4, 0.2, 1.8),
        (0.25, -3.0, 1.5, 1.0),
        (1.0, -4.0, 1.0, 1.0),
        (-5.0, -3.0, 2.0, 1.0),
        (3.0, 3.0, 1.0, 1.0),
        (-2.0, 2.0, 2.0, 1.0),
        (2.5, -0.75, 2.0, 0.5),
    ],
}

def map_size_from_bounds(bounds: dict | None) -> int:
    """根据给定的地图边界（bounds），返回对应预定义的障碍物列表。
    
    :param bounds: 根据坐标标识有效地图边界范围的字典
    :type bounds: dict | None
    """
    if bounds is None:
        return 10
    
    width = float(bounds["x_max"]) - float(bounds["x_min"])
    height = float(bounds["y_max"]) - float(bounds["y_min"])

    size = int(round(max(width, height)))

    if size not in OBSTACLES_BY_MAP:
        raise ValueError(f"No obstacle profile for map bounds {bounds}")
    return size

def obstacles_for_bounds(bounds: dict | None = None) -> list[Rectangle]:
    """
    根据边界获取maze地图预定义的障碍物信息

    :param bounds: 根据坐标标识有效地图边界范围的字典
    :type bounds: dict | None
    """
    return OBSTACLES_BY_MAP[map_size_from_bounds(bounds)]

def point_in_obstacle(
    point:Point,
    bounds: dict | None = None,
    obstacles: Iterable[Rectangle] | None = None,
) -> bool:
    """判断当前坐标点是否在障碍物内"""
    rectangles = obstacles if obstacles is not None else obstacles_for_bounds(bounds) # 如果给定了障碍物信息，就根据给定的选择，如果没有给定障碍物信息，就根据预定义的地图障碍物获取
    return any(ox <= point.x <= ox + w and oy <= point.y <= oy + h for ox, oy, w, h in rectangles)

def _is_valid(point:Point, bounds: dict | None, clearance: float)-> bool:
    """带安全间距（Clearance）的碰撞检测函数.核心作用是判断一个给定坐标点 (x, y) 在考虑了机器人/代理自身尺寸（即 clearance）后，是否处于一个合法的、无碰撞的自由空间中。
    
    :param bounds: 根据坐标标识有效地图边界范围的字典
    :type bounds: dict | None
    """

    # 边界检查：将地图收缩clearance，判断机器人中心是否在边界里面
    if bounds and not (bounds["x_min"] + clearance <= point.x <= bounds["x_max"] - clearance and bounds["y_min"] + clearance <= point.y <= bounds["y_max"] - clearance):
        return False # 如果越界了就返回False
    
    obstacles = obstacles_for_bounds(bounds)
    for ox, oy, width, height in obstacles:
        if not (
            ox - clearance < point.x < ox + width + clearance and oy - clearance < point.y < oy + height + clearance
        ):  # 当前点是否落在“障碍物向外膨胀 clearance 距离后的包围盒”内
            continue  # 不在膨胀盒内，跳过该障碍物

        # 精确计算点到矩形障碍物的最短欧氏距离，判断是否小于安全缓冲半径 clearance
        nearest_x = min(max(point.x, ox), ox + width)
        nearest_y = min(max(point.y, oy), oy + height)
        if (point.x - nearest_x) ** 2 + (point.y - nearest_y) ** 2 <= clearance ** 2:
            return False

    return True  # 通过边界检查且与所有障碍物都保持安全距离



    


def is_position_valid(
    point:Point,
    bounds: dict | None = None,
    clearance: float = 0.35
) -> bool:
    return _is_valid(point, bounds, float(clearance))