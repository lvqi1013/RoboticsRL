#!/bin/bash

# ============================================
# TurtleBot4 Gazebo 仿真启动脚本（交互式）
# ============================================

# 1. 交互式获取环境变量
read -rp "请输入 GZ_PARTITION [默认: SAC]: " input_partition
export GZ_PARTITION="${input_partition:-SAC}"

read -rp "请输入 ROS_DOMAIN_ID [默认: 10]: " input_domain_id
export ROS_DOMAIN_ID="${input_domain_id:-10}"

# 2. 交互式获取地图尺寸并映射到 world 参数
read -rp "请输入地图大小 (4/6/10) [默认: 4]: " input_mapsize
input_mapsize="${input_mapsize:-4}"

case "$input_mapsize" in
    4)  WORLD="four_maze" ;;
    6)  WORLD="six_maze"  ;;
    10) WORLD="big_maze"  ;;
    *)
        echo "❌ 错误: 不支持的地图大小 '$input_mapsize'，仅支持 4、6、10"
        return 1 2>/dev/null || exit 1
        ;;
esac

# 3. 打印确认信息
echo "========================================"
echo " GZ_PARTITION  = $GZ_PARTITION"
echo " ROS_DOMAIN_ID = $ROS_DOMAIN_ID"
echo " 地图大小      = $input_mapsize → world:=$WORLD"
echo "========================================"

# 4. 进入工作空间并启动
cd experiment2/ || { echo "❌ 目录 experiment2/ 不存在"; return 1 2>/dev/null || exit 1; }
source install/setup.bash

ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
    model:=lite \
    world:="$WORLD" \
    world_name:=maze