import numpy as np
from math import cos, sin, radians

# 定义DH建模后相邻两杆的单级齐次变换矩阵（采MDH）
def T_mat_DH(alpha, a, theta, d):
    alpha = radians(alpha)  # 将角度转换为弧度
    theta = radians(theta)   # 将角度转换为弧度
    matrix = np.mat(np.zeros((4, 4)))  # 创建4x4的零矩阵

    # 填充齐次变换矩阵
    matrix[0, 0] = cos(theta)
    matrix[0, 1] = -sin(theta)
    matrix[0, 3] = a
    matrix[1, 0] = sin(theta) * cos(alpha)
    matrix[1, 1] = cos(theta) * cos(alpha)
    matrix[1, 2] = -sin(alpha)
    matrix[1, 3] = -sin(alpha) * d
    matrix[2, 0] = sin(theta) * sin(alpha)
    matrix[2, 1] = cos(theta) * sin(alpha)
    matrix[2, 2] = cos(alpha)
    matrix[2, 3] = cos(alpha) * d
    matrix[3, 3] = 1  # 齐次坐标的最后一行
    return matrix  # 返回计算得到的齐次变换矩阵

# 输入为6x4的DH参数矩阵, 输出为从工具坐标系到基坐标系的4x4齐次变换矩阵
def DOF6_matrix(DH_parameter_matrix):
    DH_mat = DH_parameter_matrix
    DOF6_mat = np.identity(4)  # 初始化为单位矩阵
    for i in range(0, 6, 1):  # 遍历6个关节
        temp_mat = T_mat_DH(DH_mat[i, 0], DH_mat[i, 1], DH_mat[i, 2], DH_mat[i, 3])  # 计算每个关节的变换矩阵
        DOF6_mat = DOF6_mat * temp_mat  # 逐步计算最终的齐次变换矩阵
    return DOF6_mat  # 返回最终的齐次变换矩阵

def Forward_solving(observation, action):
    # 定义DH参数
    DH_alpha = np.array([0, -90, 0, -90, 90, 0])  # 每个关节的alpha角
    DH_a = np.array([0, 0, -0.425, 0, 0, 0])  # 每个关节的a长度
    Initial_theta = observation[:6]  # 每个关节的初始theta角
    DH_theta = action[:6]  # 每个关节的theta角
    DH_d = np.array([0, 0.152, 0, 0, 0.102, 0.102])  # 每个关节的d距离

    # 将以上DH参数变为一个6x4的矩阵
    DHparameter_matrix = np.mat([[DH_alpha[0], DH_a[0], Initial_theta[0] + DH_theta[0], DH_d[0]],
                                   [DH_alpha[1], DH_a[1], Initial_theta[1] + DH_theta[1], DH_d[1]],
                                   [DH_alpha[2], DH_a[2], Initial_theta[2] + DH_theta[2], DH_d[2]],
                                   [DH_alpha[3], DH_a[3], Initial_theta[3] + DH_theta[3], DH_d[3]],
                                   [DH_alpha[4], DH_a[4], Initial_theta[4] + DH_theta[4], DH_d[4]],
                                   [DH_alpha[5], DH_a[5], Initial_theta[5] + DH_theta[5], DH_d[5]]])

    # 计算最终的齐次变换矩阵
    mat = DOF6_matrix(DHparameter_matrix)
    return mat
# | R11 R12 R13 Px |
# | R21 R22 R23 Py |
# | R31 R32 R33 Pz |
# | 0   0   0   1  |