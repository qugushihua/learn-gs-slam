import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    #各参数含义
    #poses_gt所有的真实位姿数据的列表，  poses_est所有的估计位姿数据的列表，  plot_dir结果数据的保存目录
    #label评估结果的标签，  monocular：是否是单目视觉里程计
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt) #真实轨迹
    traj_est = PosePath3D(poses_se3=poses_est) #估计轨迹
    traj_est_aligned = trajectory.align_trajectory( #对齐后的估计轨迹
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part #一个枚举值，表示位姿关系
    data = (traj_ref, traj_est_aligned) #一个元组，包含了真实轨迹和对齐后的估计轨迹
    ape_metric = metrics.APE(pose_relation) #一个APE对象，用于计算绝对姿态误差
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse) #一个浮点数，表示绝对轨迹误差的均方根误差
    ape_stats = ape_metric.get_all_statistics() #一个字典，包含了所有的绝对姿态误差统计信息
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    #各参数含义：
    #frames：一个字典，包含了所有的帧（frames）数据
    #kf_ids：一个列表，包含了所有的关键帧（keyframes）的ID
    #save_dir：结果数据的保存目录，  iterations：迭代次数
    #final：是否是最后一次迭代
    #monocular：是否是单目
    trj_data = dict() #字典，用于存储轨迹（trajectory）数据
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1 #最新的帧的索引
    trj_id, trj_est, trj_gt = [], [], [] #轨迹ID、估计的位姿、真实的位姿
    trj_est_np, trj_gt_np = [], [] #它们的NumPy数组形式

    def gen_pose_matrix(R, T): #用于生成位姿矩阵
        pose = np.eye(4) #一个4x4的单位矩阵，即对角线上的元素为1，其余元素为0。
        pose[0:3, 0:3] = R.cpu().numpy() #将旋转矩阵R转换为NumPy数组，并将其赋值给pose矩阵的前3行前3列，即表示位姿的旋转部分
        pose[0:3, 3] = T.cpu().numpy() #将平移向量T转换为NumPy数组，并将其赋值给pose矩阵的前3行的第4列，即表示位姿的平移部分
        return pose

    for kf_id in kf_ids: #kf_ids列表中的每个关键帧ID
        kf = frames[kf_id] #从frames字典中获取该关键帧ID对应的关键帧数据，存储在变量kf中
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T)) #调用gen_pose_matrix函数生成估计的位姿矩阵，并求逆，得到估计位姿pose_est
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt)) #得到真实的位姿

        trj_id.append(frames[kf_id].uid) #该关键帧的ID添加到trj_id列表中
        trj_est.append(pose_est.tolist()) #估计的位姿矩阵pose_est转换为列表后加到trj_est中
        trj_gt.append(pose_gt.tolist()) #真实

        trj_est_np.append(pose_est) #np就是直接加入数组
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot") #构建一个新的目录路径，用于存储绘图数据
    mkdir_p(plot_dir) #如果该目录不存在，创建它

    label_evo = "final" if final else "{:04}".format(iterations) #根据final参数的值，设置label_evo为"final"或者格式化的迭代次数
    with open( #打开一个新的JSON文件，用于写入轨迹数据
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4) #将trj_data字典中的数据写入到JSON文件中，每个键值对之间使用4个空格进行缩进

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe, #一个管道对象，用于渲染图像
    background, #一个背景图像
    kf_indices, #关键帧索引
    iteration="final",
):
    interval = 5 #帧的间隔
    img_pred, img_gt, saved_frame_idx = [], [], [] #预测图像、真实图像和保存的帧索引
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration #结束的帧索引
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity( #一个LPIPS对象，用于计算感知图像差异
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_ssim"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
