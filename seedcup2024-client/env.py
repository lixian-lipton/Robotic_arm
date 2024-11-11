import os
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R


class Env:
    def __init__(self, is_senior, seed=114514, gui=False):
        """
        初始化环境

        参数:
        is_senior (bool): 指示是否为高级模式
        seed (int): 随机种子，用于初始化环境
        gui (bool): 是否使用图形用户界面
        """
        self.seed = seed  # 设置随机种子
        self.is_senior = is_senior  # 设置模式（高级或非高级）
        self.step_num = 0  # 当前步骤数
        self.max_steps = 100  # 最大步骤数限制

        # 初始化 PyBullet 客户端
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.setGravity(0, 0, -9.81)  # 设置重力
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加搜索路径
        self.init_env()  # 初始化环境

    def init_env(self):
        """初始化环境中的物体和设置"""
        np.random.seed(self.seed)  # 设置随机种子
        # 加载 FR5 机器人模型
        self.fr5 = self.p.loadURDF("fr5_description/urdf/fr5v6.urdf", useFixedBase=True, basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]),
                                   flags=p.URDF_USE_SELF_COLLISION)
        # 加载桌子模型
        self.table = self.p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                     baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        # 创建目标物体
        collision_target_id = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_target_id,
                                             basePosition=[0.5, 0.8, 2])
        # 创建障碍物
        collision_obstacle_id = self.p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
        self.obstacle1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_obstacle_id,
                                                basePosition=[0.5, 0.5, 2])
        self.reset()  # 重置环境状态

    def reset(self):
        """重置环境状态"""
        self.step_num = 0  # 重置步骤计数
        self.success_reward = 0  # 重置成功奖励
        self.terminated = False  # 结束标志
        self.obstacle_contact = False  # 障碍物接触标志
        # 设置机器人关节的中立角度
        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118,
                         -49.45849125928217, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]  # 转换为弧度
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                         targetPositions=neutral_angle)
        # 随机生成目标位置
        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        # 更新目标物体的位置
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])
        # 随机生成障碍物的位置
        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])
        # 进行一定数量的仿真步进以稳定环境
        for _ in range(100):
            self.p.stepSimulation()
        return self.get_observation()  # 返回初始观测值

    def get_observation(self):
        """获取当前环境的观测值"""
        # 获取机器人的关节角度，并转换为度数
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 7)]
        # 归一化关节角度
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        # 获取目标和障碍物的位置
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        # 将观测值组合成一个一维数组
        self.observation = np.hstack((obs_joint_angles, target_position, obstacle1_position)).flatten().reshape(1, -1)
        return self.observation  # 返回观测值

    def step(self, action):
        """执行指定动作并返回新的状态、奖励和结束标志"""
        if self.terminated:
            return self.reset_episode()  # 如果已结束，重置回合

        self.step_num += 1  # 增加步骤计数

        # 获取当前关节角度
        joint_angles = [self.p.getJointState(self.fr5, i)[0] for i in range(1, 7)]
        action = np.clip(action, -1, 1)  # 限制动作范围
        fr5_joint_angles = np.array(joint_angles) + (np.array(action[:6]) / 180 * np.pi)  # 更新关节角度
        gripper = np.array([0, 0])  # 夹爪状态（未使用）
        angle_now = np.hstack([fr5_joint_angles, gripper])  # 合并关节角度和夹爪状态

        self.reward()  # 计算奖励
        # 应用关节控制
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL,
                                         targetPositions=angle_now)

        # 进行多步仿真
        for _ in range(20):
            self.p.stepSimulation()

        return self.observation  # 返回新的观测值

    def get_dis(self):
        """计算夹爪中心与目标之间的距离"""
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]  # 获取夹爪位置
        relative_position = np.array([0, 0, 0.15])  # 定义相对位置
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])  # 获取夹爪的旋转状态
        rotated_relative_position = rotation.apply(relative_position)  # 应用旋转
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position  # 计算夹爪中心位置

        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])  # 获取目标位置
        return np.linalg.norm(gripper_centre_pos - target_position)  # 返回距离

    def reward(self):
        """计算当前奖励并检查接触状态"""
        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]  # 获取接触的链接索引
            if link_index not in [0, 1]:  # 如果不是底盘或底座
                self.obstacle_contact = True  # 标记为接触状态

        # 计算奖励
        if self.get_dis() < 0.05 and self.step_num <= self.max_steps:  # 如果距离目标很近
            self.success_reward = 100  # 成功奖励
            if self.obstacle_contact:  # 如果接触了障碍物
                if self.is_senior:
                    self.success_reward = 20  # 高级模式的奖励
                elif not self.is_senior:
                    self.success_reward = 50  # 非高级模式的奖励
                else:
                    return  # 不执行任何操作
            self.terminated = True  # 设置为结束状态

        elif self.step_num >= self.max_steps:  # 如果达到最大步骤数
            distance = self.get_dis()  # 计算当前距离
            if 0.05 <= distance <= 0.2:  # 如果在特定范围内
                self.success_reward = 100 * (1 - ((distance - 0.05) / 0.15))  # 根据距离计算奖励
            else:
                self.success_reward = 0  # 超出范围，奖励为零
            if self.obstacle_contact:  # 如果接触了障碍物
                if self.is_senior:
                    self.success_reward *= 0.2  # 高级模式的惩罚
                elif not self.is_senior:
                    self.success_reward *= 0.5  # 非高级模式的惩罚

            self.terminated = True  # 设置为结束状态

    def reset_episode(self):
        """重置回合并返回步骤数和距离"""
        self.reset()  # 重置环境
        return self.step_num, self.get_dis()  # 返回步骤数和当前距离

    def close(self):
        """关闭环境并释放资源"""
        self.p.disconnect()  # 断开与 PyBullet 的连接