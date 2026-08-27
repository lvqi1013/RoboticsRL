# -*- coding:utf-8 -*-
"""
@file name  : move_marker.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/23
@brief      : 基于{spawn_marker.py}创建的标记，移动这个标记。
"""


from gz.transport14 import Node as GzNode
from gz.msgs11.entity_factory_pb2 import EntityFactory
from gz.msgs11.boolean_pb2 import Boolean
from gz.msgs11.pose_pb2 import Pose as GzPose

def move_marker(name, position):
    pose_msg = GzPose()
    pose_msg.name = name
    pose_msg.position.x = float(position[0])
    pose_msg.position.y = float(position[1])
    pose_msg.position.z = 0.001
    pose_msg.orientation.w = 1.0

    service_name = "/world/maze/set_pose"
    gz_node = GzNode()
    try:
        gz_node.request(service_name, pose_msg, GzPose, Boolean, 300)
    except Exception as e:
        print(f"Failed to move marker {name}: {e}")

if __name__ == '__main__':
    move_marker('marker_visual', position=[1.5, 1.5])