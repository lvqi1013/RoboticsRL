import os
import argparse
from nav_code_rl.nav_node import TurtleBotRLNode
import rclpy

from configs.common_config import MAP_BOUNDS, OBSTACLES

MIN_DISTANCE = {4: 2.0, 6: 4.0, 10: 7.0}

class Args:
    map_size: int
    algorithm: str
    timesteps: int
    episodes: int
    seed: int
    eval_only: bool
    subgoal_mode: str

def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-size", type=int, default=4, choices=[4, 6, 10])
    parser.add_argument('--algorithm', type=str, default='PPO',choices=['PPO', 'SAC'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--eval-only", type=bool, default=False)
    parser.add_argument("--subgoal_mode", default='none', choices=['none', 'model', 'rule'])
    return parser.parse_args()

def main():
    args = get_args()
    rclpy.init()

    map_size = args.map_size
    bounds = MAP_BOUNDS[map_size]
    obstacles = OBSTACLES[map_size]
    
    pos_file_dir = 'experiment2/results/position_start_goal'
    pos_file = os.path.join(pos_file_dir, f'position_mapsize_{map_size}.json')

    node = TurtleBotRLNode(map_bounds=bounds,
                           obstacles=obstacles,
                           algorithm=args.algorithm,
                           seed=args.seed,
                           timesteps=args.timesteps,
                           episodes=args.episodes,
                           min_distance=MIN_DISTANCE[map_size],
                           eval_only=args.eval_only,
                           subgoal_mode=args.subgoal_mode,
                           positions_file=pos_file
                           )
    
    node.train_and_evaluate()
    node.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()