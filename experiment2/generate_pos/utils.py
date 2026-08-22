from configs.common_config import DEFAULT_CLEARANCE

def point_in_obstacle(x, y, obstacles):
    """
    判断点 (x, y) 是否在任意一个障碍物内。
    返回 True 表示在障碍物内，False 表示不在。
    """
    for ox, oy, w, h in obstacles:
        if ox <= x <= ox + w and oy <= y <= oy + h:
            return True
    return False

def is_spawn_position_valid(x, y, obstacles, bounds=None, clearance=DEFAULT_CLEARANCE,):
    """
    判断一个点能否作为起/终点，需同时满足：

    距地图边界至少 clearance（0.51m）
    不在任何障碍物矩形内
    距任一障碍物表面距离 ≥ clearance（障碍物外扩 0.51m 的碰撞安全区）
    """
    # 判断是否在合法地图内
    if bounds:
        if not (bounds['x_min'] + clearance <= x <= bounds['x_max'] - clearance and
                bounds['y_min'] + clearance <= y <= bounds['y_max'] - clearance):
            return False

    # 判断是否在障碍物里
    if point_in_obstacle(x, y, obstacles):
        return False

    # 判断距离障碍物表面是否合法
    for ox, oy, w, h in obstacles:
        expanded_left = ox - clearance
        expanded_right = ox + w + clearance
        expanded_bottom = oy - clearance
        expanded_top = oy + h + clearance

        if expanded_left <= x <= expanded_right and expanded_bottom <= y <= expanded_top:
            nearest_x = min(max(x, ox), ox + w)
            nearest_y = min(max(y, oy), oy + h)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= clearance * clearance:
                return False

    return True

