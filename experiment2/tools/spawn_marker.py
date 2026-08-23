# -*- coding:utf-8 -*-
"""
@file name  : spawn_marker.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/23
@brief      : 通过python端直接调用与Gazebo交互的接口测试生成一个仅有视觉标记的点。
"""

from gz.transport14 import Node as GzNode
from gz.msgs11.entity_factory_pb2 import EntityFactory
from gz.msgs11.boolean_pb2 import Boolean

def spawn_marker(name, position, color="1 0 0 1"):
    """在Gazebo中通过Gazebo实体生成一个圆柱体标志，仅视觉显示，没有碰撞体，用于标识起点和终点"""
    sdf_string = f"""
    <?xml version="1.0" ?>
    <sdf version="1.6">
        <model name="{name}">
            <static>true</static>
            <link name="link">
                <visual name="visual">
                    <geometry>
                        <cylinder>
                            <radius>0.2</radius>
                            <length>0.01</length>
                        </cylinder>
                    </geometry>
                    <material>
                        <ambient>{color}</ambient>
                        <diffuse>{color}</diffuse>
                        <specular>0 0 0 1</specular>
                    </material>
                </visual>
            </link>
        </model>
    </sdf>
    """

    req = EntityFactory()
    req.sdf = sdf_string
    req.pose.position.x = float(position[0])
    req.pose.position.y = float(position[1])
    req.pose.position.z = 0.001    

    service_name = "/world/maze/create"

    gz_node = GzNode()
    gz_node.request(service_name, req, EntityFactory, Boolean, 1000)


if __name__ == '__main__':
    spawn_marker(name='marker_visual', position=[0.0, 0.0])
