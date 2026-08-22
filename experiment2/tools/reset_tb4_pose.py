# -*- coding:utf-8 -*-
"""
@file name  : reset_tb4_pose.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/21
@brief      : 通过 Gazebo Transport (gz-transport) 协议，直接调用仿真世界的服务接口，将 TurtleBot4 机器人重置到原点位置，并朝向x轴正方向。同步调用（一直等待直到收到响应或达到 timeout_ms 超时）。异步方式是采用回调
"""

from gz.msgs11.pose_pb2 import Pose as GzPose
from gz.msgs11.boolean_pb2 import Boolean
from gz.transport14 import Node as GzNode
import math

def reset_turtlebot4_position():
    gznode = GzNode()

    pose_msg: GzPose = GzPose()
    pose_msg.name = 'turtlebot4'
    pose_msg.position.x = 0.0
    pose_msg.position.y = 0.0
    pose_msg.position.z = 0.0

    yaw = 0.0
    pose_msg.orientation.w = math.cos(yaw / 2.0)
    pose_msg.orientation.x = 0.0
    pose_msg.orientation.y = 0.0
    pose_msg.orientation.z = math.sin(yaw / 2.0)

    service_name = "/world/maze/set_pose"
    timeout_ms = 1000

    result, response = gznode.request(service=service_name, request=pose_msg, request_type=GzPose, response_type=Boolean, timeout=timeout_ms)
    if result and response and response.data:
        print("Robot position reset successfully")

if __name__ == '__main__':
    reset_turtlebot4_position()
