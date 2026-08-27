import time
import rclpy
from nav_msgs.msg import Odometry
import tf_transformations
import numpy as np

class ResetOdom:
    def __init__(self):
        rclpy.init()
        self.odom_seq = 0
        self.max_wait_for_observation = 5.0 
        self.node = rclpy.create_node('turtlebot_nav_env_odom')
        self.odom_sub = self.node.create_subscription(msg_type=Odometry, topic='/odom', callback=self.odom_callback, qos_profile=10)

        self.current_position = [0.0 , 0.0]

    def wait_for_odom_updates(self, num_updates=1):
        start_time = time.time()
        target_seq = self.odom_seq + num_updates
        while (self.odom_seq < target_seq) and (time.time() - start_time) < self.max_wait_for_observation:
            # 当更新序号未达到目标序号，或者当前时间没有超过最开始的时间的时候，执行更新
            rclpy.spin_once(self.node, timeout_sec=0.05)
        
        if self.odom_seq < target_seq:
            # 如果退出循环后仍未达到目标序列号，说明发生了超时，进入安全回退逻辑：
            print(f"Warning: Odom reception timed out (received {self.odom_seq - (target_seq - num_updates)}/{num_updates}). Using fallback pose.")
            if self.current_position is None:
                self.current_position = self.start_position.copy()
            if not np.isfinite(self.current_yaw):
                self.current_yaw = 0.0
            self.current_yaw = float(getattr(self, 'last_reset_yaw', 0.0))

    def odom_callback(self, msg):
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            self.current_position = np.array([pos.x, pos.y], dtype=np.float32)
            _, _, yaw = tf_transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            self.current_yaw = float(yaw)
            self.odom_seq += 1
            print(pos)
        except Exception:
            pass

if __name__ == '__main__':
    resodom = ResetOdom()
    resodom.wait_for_odom_updates(5)
    print(resodom.odom_seq)