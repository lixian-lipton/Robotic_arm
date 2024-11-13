import os
import numpy as np

from stable_baselines3 import PPO
from self_solving import MyMethod
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod 
    def get_action(self, observation):
        """
        输入观测值，返回动作
        Args:
            observation: numpy array of shape (1, 12) 包含:
                - 6个关节角度 (归一化到[0,1])
                - 3个目标位置坐标
                - 3个障碍物位置坐标
        Returns:
            action: numpy array of shape (6,) 范围在[-1,1]之间
        """
        pass

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        # 自定义初始化
        pass
        
    def get_action(self, observation):
        # 输入观测值，返回动作
        observation = observation.flatten()
        action = MyMethod(observation)
        return action

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.zip")
        self.model = PPO.load(model_path, device="cuda") #############remember to change to "cpu" before submitting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def get_action(self, observation):
        action, _ = self.model.predict(observation)
        action = action.flatten()
        return action
    

