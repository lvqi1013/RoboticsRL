#!/bin/bash

# 激活虚拟环境
source .venv/bin/activate

# 设置 PYTHONPATH
export PYTHONPATH=experiment2/:$PYTHONPATH

# 交互式获取 GZ_PARTITION（默认值为 SAC）
read -rp "请输入 GZ_PARTITION [默认: SAC]: " input_partition
export GZ_PARTITION="${input_partition:-SAC}"

# 交互式获取 ROS_DOMAIN_ID（默认值为 10）
read -rp "请输入 ROS_DOMAIN_ID [默认: 10]: " input_domain_id
export ROS_DOMAIN_ID="${input_domain_id:-10}"

# 打印确认信息
echo "----------------------------------------"
echo "GZ_PARTITION   = $GZ_PARTITION"
echo "ROS_DOMAIN_ID  = $ROS_DOMAIN_ID"
echo "PYTHONPATH     = $PYTHONPATH"
echo "----------------------------------------"