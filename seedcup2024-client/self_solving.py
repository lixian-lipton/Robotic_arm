import numpy as np
from robot_arm import Forward_solving

def get_gripper_pos(obs, act):
    mat = Forward_solving(obs, act)
    gripper_position = np.array([mat[3, 3], mat[0, 3], mat[1, 3]])
    return gripper_position

def MyMethod(observation):
    robot_state = observation[:6]
    target_position = observation[6:9]
    obstacle_position = observation[9:]
    ini_robot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    gripper_position = get_gripper_pos(ini_robot, observation)
    if target_position[0] > gripper_position[0]:
        action[0] = -1
    else:
        action[0] = 1

    return action
    # 右 左 上 上 顺 逆
