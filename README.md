# 安装 turtlebot 仿真环境

```bash
sudo apt install ros-jazzy-turtlebot4-simulator ros-jazzy-irobot-create-nodes -y
sudo apt install ros-dev-tools
```

# 安装 Gazebo 开发环境 Python 包

```bash
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt update
sudo apt install -y libgz-transport14-dev python3-gz-transport14
```

# 全局环境的包

```bash
sudo apt install libgz-msgs11-dev python3-gz-msgs11
sudo apt install -y ros-${ROS_DISTRO}-tf-transformations
```

# 配置虚拟环境

```bash
uv venv --system-site-packages --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

# 启动仿真地图

```bash
# GUI
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py model:=lite world:=maze

# without GUI
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py model:=lite world:=maze headless:=true
```
