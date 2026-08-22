import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from utils import is_spawn_position_valid
from configs.common_config import MAP_BOUNDS, OBSTACLES,MIN_DISTANCE, EXPERIMENT2_RESULTS_PATH

class GPArgs:
    map_size:int

def get_GPargs()-> GPArgs:
    parser  = argparse.ArgumentParser()
    parser.add_argument("--map-size", type=int,default=4)
    return parser.parse_args()

def generate_random_positions(map_bounds, min_distance, obstacles):
    """
    generate_random_positions 的 Docstring
    
    :param map_bounds: 地图障碍物定义
    :param min_distance: 起点和终点的最短欧氏距离定义
    """
    max_attempts = 10000

    for i in range(max_attempts):
        start_x = round(np.random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2) # 在地图的x的范围上随机采样一个点并且保留两位小数
        start_y = round(np.random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)

        # 如果不能作为起点和终点则继续下一轮采集
        if not is_spawn_position_valid(start_x, start_y,obstacles, bounds=map_bounds):
            continue   

        goal_x = round(np.random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
        goal_y = round(np.random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
        if not is_spawn_position_valid(goal_x, goal_y, obstacles, bounds=map_bounds):
            continue     

        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if distance >= min_distance:
            return [start_x, start_y], [goal_x, goal_y]
        
    # 如果无法生成有效位置，使用默认值
    return [-1.4, 1.0], [0.4, -1.4]      

def main():
    args = get_GPargs()

    map_size = args.map_size
    pairs = []
    NUM_PAIRS = 100000
    for i in tqdm(range(NUM_PAIRS), desc = 'generate positions', ncols = 80):
        start, goal = generate_random_positions(map_bounds=MAP_BOUNDS[map_size],obstacles=OBSTACLES[map_size],min_distance=MIN_DISTANCE)
        pairs.append({'start': start, 'goal': goal})
    
    pos_start_goal_path = EXPERIMENT2_RESULTS_PATH / "position_start_goal"
    pos_start_goal_path.mkdir(parents=True, exist_ok=True)

    out_json = pos_start_goal_path / f"position_mapsize_{map_size}.json"

    # 保存为JSON文件
    with open(out_json, 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"已生成{NUM_PAIRS}对起终点，保存至{out_json.absolute()}")

if __name__ == '__main__':
    main()
