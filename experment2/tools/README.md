# 文档说明

- 本文件夹用于存放 Gazebo 相关的文档和代码示例
- 文档以 markdown 格式编写，代码示例以 python 格式编写

[reset_tb4_pose.py](./reset_tb4_pose.py): 该文档存放重置 TurtleBot4 的位姿的示例代码
[check_gz_msg_pose.py](./check_gz_msg_pose.py): 该文档存放检查 Gazebo 消息中的位姿的示例代码

# 查看 Gazebo 中的服务

查看所有服务名

```bash
gz service -l
```

## 查看服务所需的数据类型

```bash
gz service -s <具体的服务名称> -i
gz service -s /world/maze/state -i
```

## 查看具体的数据类型实现

```python
def print_proto_fields(descriptor, indent=0):
    """递归打印 Protobuf 消息的所有字段"""
    prefix = "  " * indent
    for field in descriptor.fields:
        print(f"{prefix}├── {field.name} (type={field.type}, number={field.number})")
        # 如果是嵌套消息类型(11)，递归展开
        if field.type == 11 and field.message_type:
            print_proto_fields(field.message_type, indent + 1)

# 示例使用
# print_proto_fields(YourMessageType.descriptor)
```
