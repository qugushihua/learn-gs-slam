import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0) #将相机的平移增量和旋转增量拼接成一个向量 tau，代表从当前姿态到新姿态的变化量

    T_w2c = torch.eye(4, device=tau.device) #一个4x4的单位矩阵 T_w2c
    #用相机当前的旋转矩阵和平移向量来更新这个矩阵，从而得到当前的世界到相机的变换矩阵
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c #SE3_exp(tau)：将tau表示的李代数映射到SE(3)群的李群，即从旋转和平移的增量计算出实际的变换矩阵
                                   #这个变换矩阵与当前的变换矩阵 T_w2c 相乘，得到更新后的变换矩阵


    #从新的变换矩阵 new_w2c 中提取出更新后的旋转矩阵 new_R 和平移向量 new_T
    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold #计算 tau 的范数（即增量的大小）并与一个预设的阈值比较，来判断姿态更新是否已经足够小，即是否收敛。
                                                 # 如果 tau 的范数小于阈值，则认为更新已经收敛，函数返回 True；否则，返回 False
    camera.update_RT(new_R, new_T) #更新相机的姿态

    #将相机的平移增量和旋转增量重置为0，为下一次更新做准备
    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
