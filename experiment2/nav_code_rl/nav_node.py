import rclpy
from rclpy.node import Node
from env.nav_env_rule_based import TurtleBotNavEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import numpy as np
import argparse
import os
from datetime import datetime
import torch
from turtlebot4_rl.custom_callback import SuccessInfoCallback
from typing import NamedTuple

class TurtleBotRLNode(Node):
    """ROS2节点：可训练或仅评估强化学习模型。"""

    def __init__(
        self,
        map_bounds,
        obstacles,
        algorithm='PPO',
        seed = 42,
        timesteps=10000,
        episodes=10,
        model_path=None,
        min_distance=4,
        eval_only=False,
        eval_start_index=None,
        subgoal_mode: str = 'none',
        subgoal_model_path = None,
        positions_file = None
    ):
        
        super().__init__('turtlebot_rl_node')

        self.algorithm = algorithm.upper() # 使用的RL算法
        self.timesteps = timesteps
        self.episodes = episodes

        self.model_path = model_path
        self.min_distance = min_distance
        self.eval_only = eval_only  # 若为True，只进行评估不训练
        self.eval_start_index = eval_start_index  # 评估时起始 positions 索引
        self.subgoal_mode = subgoal_mode.strip().lower()

        self.map_bounds = map_bounds

        self.model_dir = os.path.join('models', self.algorithm)
        os.makedirs(self.model_dir, exist_ok=True)

        # Setup Tensorboard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_log = os.path.join('tensorboard_logs', self.algorithm, timestamp)
        os.makedirs(self.tensorboard_log, exist_ok=True)

        self.env = TurtleBotNavEnv(
            max_wait_for_observation=5.0,
            map_bounds=map_bounds,
            obstacles=obstacles,
            min_distance=min_distance,
            positions_file=positions_file,
            subgoal_mode=self.subgoal_mode,
            subgoal_model_path=subgoal_model_path,
            seed=seed
        )

        self.model = self._load_algorithm(self.algorithm, self.model_path)

        self.get_logger().info(
            f"Algorithm: {self.algorithm}, Timesteps: {self.timesteps}, Episodes: {self.episodes}, Model Path: {self.model_path}"
        )

        self.get_logger().info(f"Tensorboard logs will be saved to: {os.path.abspath(self.tensorboard_log)}")
        self.get_logger().info("To view training progress, run: tensorboard --logdir tensorboard_logs")        


    def _load_algorithm(self, algorithm_name, model_path):
        """Load or initialize the RL model based on the specified algorithm."""
        algorithms = {
            'PPO': PPO,
            'SAC': SAC,
        }
        if algorithm_name not in algorithms:
            self.get_logger().error(f"Algorithm {algorithm_name} is not supported!")
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        if model_path and os.path.isfile(model_path):
            self.get_logger().info(f"Loading pre-trained model from {model_path}")
            model = algorithms[algorithm_name].load(model_path, env=self.env, tensorboard_log=self.tensorboard_log)
        else:
            if model_path:
                self.get_logger().warning(f"Model path {model_path} not found. Initializing a new model.")
            # 为PPO添加更稳定的超参数
            if algorithm_name == 'PPO':
                model = algorithms[algorithm_name](
                    "MlpPolicy", 
                    self.env, 
                    verbose=1,
                    device='auto',
                    tensorboard_log=self.tensorboard_log,
                    learning_rate=1e-4,  
                    n_steps=2048,  
                    batch_size=256, 
                    n_epochs=10,
                    gamma=0.95,
                    gae_lambda=0.95,
                    clip_range=0.1,
                    clip_range_vf=None,
                    ent_coef=0.05,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(
                        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                        activation_fn=torch.nn.ReLU,
                        ortho_init=True,  # 使用正交初始化，提高训练稳定性
                    ),
                    normalize_advantage=True,  # 归一化优势函数，提高训练稳定性
                    target_kl=0.005,  # 限制策略更新幅度，提高稳定性
                )
            elif algorithm_name == 'SAC':
                action_dim = float(np.prod(self.env.action_space.shape)) if hasattr(self.env.action_space, "shape") else 1.0
                model = algorithms[algorithm_name](
                    "MlpPolicy",
                    self.env,
                    verbose=1,
                    device='auto',
                    tensorboard_log=self.tensorboard_log,
                    learning_rate=1e-4,
                    buffer_size=1000_000,
                    batch_size=256,
                    gamma=0.99,
                    tau=0.005,
                    train_freq=1,
                    gradient_steps=1,
                    learning_starts=5000,
                    ent_coef='auto',
                    target_entropy=-1.5*action_dim,
                    policy_kwargs=dict(
                        net_arch=[256, 256],
                        activation_fn=torch.nn.ReLU
                    )
                )

        return model        
    
    def evaluate_model(self, deterministic: bool = True):
        self.get_logger().info(f"Starting evaluation for {self.episodes} episodes (deterministic={deterministic})")
        # 如果用户指定了评估起始索引，并且环境已加载 positions 列表
        if self.eval_start_index is not None:
            if hasattr(self.env, 'positions') and self.env.positions is not None:
                # 确保不越界
                if 0 <= self.eval_start_index < len(self.env.positions):
                    self.env.position_index = self.eval_start_index
                    self.get_logger().info(f"Evaluation will begin from positions index {self.eval_start_index} (共 {len(self.env.positions)} 对)。")
                else:
                    self.get_logger().warning(f"指定的 eval_start_index={self.eval_start_index} 越界（0~{len(self.env.positions)-1}），忽略该设置。")
            else:
                self.get_logger().warning("环境未加载 positions_6.json，eval_start_index 设置被忽略，将使用随机起终点。")
        success_count = 0 
        for episode in range(1, self.episodes + 1):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                total_reward += reward
                step_count += 1
            if 'is_success' in info and info['is_success']:
                success_count += 1
            self.get_logger().info(f"[Eval] Episode {episode}: steps={step_count}, total_reward={total_reward:.3f}")
        self.get_logger().info(f"Evaluation finished. Success_rate: {success_count}/{self.episodes}")

    def train_and_evaluate(self):
        if self.eval_only:
            self.get_logger().info("Evaluation-only 模式：跳过训练，直接评估已加载模型。")
            self.evaluate_model(deterministic=True)
            return

        # 计算总训练步数
        total_timesteps = self.episodes * self.timesteps
        self.get_logger().info(f"Starting training with {total_timesteps:,} total timesteps")
        self.get_logger().info("Environment will auto-generate random start/goal positions on each reset")

        callbacks = [SuccessInfoCallback(tensorboard_log_dir=self.tensorboard_log, verbose=1)]

        training_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=callbacks
            )
            self.get_logger().info("Training completed!")
        except KeyboardInterrupt:
            self.get_logger().info("Training interrupted by user")

        final_model_path = os.path.join(
            self.model_dir,
            f"{self.algorithm}_{training_session}_FINAL_{total_timesteps}steps.zip"
        )
        self.model.save(final_model_path)
        self.get_logger().info(f"Model saved: {final_model_path}")
        self.evaluate_model(deterministic=True)


    def close(self):
        self.env.close()
        self.get_logger().info("Environment closed.")
        self.get_logger().info(f"To view Tensorboard logs, run:")
        self.get_logger().info(f"tensorboard --logdir {os.path.abspath('tensorboard_logs')}")
        self.get_logger().info("Then open http://localhost:6006 in your browser")