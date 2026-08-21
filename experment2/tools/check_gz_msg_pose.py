# -*- coding:utf-8 -*-
"""
@file name  : check_gz_msg_pose.py
@author     : Qi Lv (https://lvqi1013.github.io/)
@Email		: lvqi@hunnu.edu.cn
@date       : 2026/08/21
@brief      : 查看gz.msgs.Pose的数据结构定义
"""

from gz.msgs11.pose_pb2 import Pose
from gz.msgs11.boolean_pb2 import Boolean

def print_proto_fields(descriptor, indent=0):
    """递归打印 Protobuf 消息的所有字段"""
    prefix = "  " * indent
    for field in descriptor.fields:
        print(f"{prefix}├── {field.name} (type={field.type}, number={field.number})")
        # 如果是嵌套消息类型(11)，递归展开
        if field.type == 11 and field.message_type:
            print_proto_fields(field.message_type, indent + 1)

print("=== gz.msgs.Pose 完整结构 ===")
print_proto_fields(Pose.DESCRIPTOR)
print_proto_fields(Boolean.DESCRIPTOR)