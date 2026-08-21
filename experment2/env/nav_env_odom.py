from rclpy.node import Node

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gz.transport14 import Node as GzNode
from gz.msgs11.pose_pb2 import Pose as GzPose
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.entity_factory_pb2 import EntityFactory
import time
import math
import tf_transformations
import os

GOAL_REACH_THRESHOLD: float = 0.1  # 目标到达阈值（米）

class TurtleBotNavEnv(gym.Env):
    """Navigation environment using odometry for pose and LiDAR for perception."""

    def __init__(self, max_wait_for_observation=5.0, map_bounds=None, min_distance=2.0, positions_file=None):
        super().__init__()

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node: Node = rclpy.create_node('turtlebot_nav_env_odom')

        # 地图边界与起终点配置
        self.map_bounds = map_bounds if map_bounds  else {
            'x_min': -2, 'x_max': 2, 'y_min': -2, 'y_max': 2
        }        

        self.min_distance: float = min_distance

        # 预定义起终点文件
        self.positions = None
        self.position_index = 0



    def _print_and_log(self, message):
        self.node.get_logger().info(message)