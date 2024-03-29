#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear)
    return P

#用于生成投影矩阵
def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    #各参数：近平面、远平面、光心 x 坐标、光心 y 坐标、焦距 x、焦距 y、图像宽度和图像高度
    #根据输入的参数计算了视景体的左右、上下边界
    left = ((2 * cx - W) / W - 1.0) * W / 2.0 #光心 x 坐标相对于图像宽度的位置，将其映射到范围[-1, 1]，再乘以图像宽度的一半，得到左边界的位置
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left #根据近平面和焦距 x 对左边界进行缩放
    right = znear / fx * right #下面同理
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4) #创建一个4x4的零矩阵 P 用于存储投影矩阵

    z_sign = 1.0 #设定一个表示深度方向的符号。符号为正时，表示深度方向是指从相机向外延伸的方向；当这个符号为负时，表示深度方向是指从相机向内收缩的方向

    P[0, 0] = 2.0 * znear / (right - left) #沿x轴缩放坐标
    P[1, 1] = 2.0 * znear / (top - bottom) #沿y轴缩放坐标
    P[0, 2] = (right + left) / (right - left) #沿x轴平移坐标
    P[1, 2] = (top + bottom) / (top - bottom) #沿y轴平移坐标
    P[3, 2] = z_sign #控制深度值的方向
    P[2, 2] = z_sign * zfar / (zfar - znear) #沿z轴缩放坐标
    P[2, 3] = -(zfar * znear) / (zfar - znear) #沿z轴平移坐标
    # fmx  0    px   0
    # 0    fmy  py   0
    # 0    0    fmz  -pz
    # 0    0    1    0

    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
