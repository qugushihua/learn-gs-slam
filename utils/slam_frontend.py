import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True #系统是否要重置，初始化后设置为 False
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1 #每隔多少帧使用一次

        self.gaussians = None
        self.cameras = dict() #保存所有的相机视角
        self.device = "cuda:0"
        self.pause = False

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"] #从配置中获取 RGB 边界阈值
        # 将当前帧索引添加到关键帧索引列表中
        self.kf_indices.append(cur_frame_idx)
        # 获取当前帧的视角信息
        viewpoint = self.cameras[cur_frame_idx]
        # 获取当前视角的原始图像 gt_img
        gt_img = viewpoint.original_image.cuda()
        # 计算图像中 RGB 像素值的和，然后与 RGB 边界阈值进行比较，得到有效的 RGB 像素值
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:#如果系统是单目系统
            if depth is None: #如果不提供深度图
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])# 初始化深度为一个固定值，并添加一些噪声
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else: #提供了深度图
                # 根据深度和不透明度信息计算初始深度，并根据有效 RGB 像素值进行掩码
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:#如果使用逆深度，上面给定为false
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth( #计算逆深度的中值、标准差以及有效掩码
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or( #根据逆深度、中值和标准差的关系，确定哪些逆深度值是无效的，即超出中值加减标准差范围的逆深度值。这些值被标记为无效，并在 invalid_depth_mask 中记录
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or( #将无效逆深度掩码与有效的 RGB 像素值的掩码取并集，以确保无效的逆深度值不会影响有效像素的深度计算
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth #将无效的逆深度值替换为逆深度的中值，以修正这些无效值
                    inv_initial_depth = inv_depth + torch.randn_like( #对修正后的逆深度值添加服从标准正态分布的噪声
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth #将修正和添加噪声后的逆深度值转换回深度值，得到最终的初始深度图 initial_depth。这一步是因为逆深度是深度的倒数，所以需要将其倒数再转换回深度值
                else: #不使用逆深度
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0] #将深度数据转换为 NumPy 数组并返回
        # use the observed depth 若不是单目，那么就使用观测到的深度信息
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0) #从视角中获取深度信息，并将其转换为张量
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels 忽略无效的 RGB 像素值，并将它们的深度设置为零
        return initial_depth[0].numpy() #将深度数据转换为 NumPy 数组并返回

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular #如果系统不是单目（monocular），则将初始化状态设置为 True，否则为 False
        self.kf_indices = [] #初始化关键帧索引列表为空
        self.iteration_count = 0 #初始化迭代计数器为 0
        self.occ_aware_visibility = {} #初始化一个空字典，用于存储每个关键帧的可见性信息
        self.current_window = [] #初始化当前窗口为空，该窗口用于跟踪一系列关键帧
        # remove everything from the queues 清空队列中的数据：通过一个循环清空后端队列中的所有数据，以确保系统处于干净的状态
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        # 将当前视角的旋转和平移更新为gt真实姿态。(并放到gpu上)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = [] #再次将关键帧索引列表清空
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True) # 添加新的关键帧，并生成深度地图
        self.request_init(cur_frame_idx, viewpoint, depth_map)  #向后端请求初始化，传递当前帧索引、视角信息和深度地图
        self.reset = False #重置标志位为 False，表示系统不需要重置

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames] #获取当前处理帧之前的一个特定帧作为 prev，当前帧的索引-在处理帧时跳过的帧数
        viewpoint.update_RT(prev.R, prev.T) #更新当前帧的旋转和平移参数，并转移它们到当前设备上

        opt_params = [] #创建一个优化参数列表 opt_params
        #当前视角的旋转增量
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        #平移增量
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        #曝光参数
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        # 初始化一个 Adam 优化器，用于优化
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            #使用 render 函数渲染当前视角
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            #获取渲染的图像、深度和不透明度
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            pose_optimizer.zero_grad() #梯度清零
            # 计算跟踪过程中的损失函数
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward() #反向传播，计算梯度

            with torch.no_grad(): #接下来的代码块将不计算梯度，这是因为在更新模型参数时不需要计算梯度
                pose_optimizer.step() #调用优化器的 step 方法来更新 viewpoint 的姿态参数
                converged = update_pose(viewpoint) #更新 viewpoint 的姿态，并检查是否已经收敛；如果已经收敛，converged 将被设置为 True，表示当前的视角已经足够接近目标，不需要进一步的优化

            if tracking_itr % 10 == 0: #每隔一定次数的迭代，将当前的视角信息发送到可视化队列 self.q_main2vis 中，以便进行可视化
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        #从配置中获取关于关键帧的参数
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx] #当前帧的相机姿态信息
        last_kf = self.cameras[last_keyframe_idx] #上一个关键帧的相机姿态信息
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T) #当前帧相机姿态相对于世界坐标系的变换矩阵
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T) #上一个关键帧相机姿态相对于世界坐标系的变换矩阵
        last_kf_WC = torch.linalg.inv(last_kf_CW) #求逆
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3]) #当前帧与上一个关键帧之间的距离
        dist_check = dist > kf_translation * self.median_depth #检查距离是否大于设定的距离阈值乘以中值深度
        dist_check2 = dist > kf_min_translation * self.median_depth #检查距离是否大于设定的最小距离阈值乘以中值深度

        union = torch.logical_or( #当前帧可见性和上一个关键帧可见性的并集中非零元素的数量
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and( #当前帧可见性和上一个关键帧可见性的交集中非零元素的数量
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union #点云匹配比率

        #判断当前帧是否应该成为一个新的关键帧。
        #条件为：点云匹配比率小于设定的重叠阈值 kf_overlap 且距离符合最小距离条件 dist_check2，或者距离符合距离阈值条件 dist_check
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        #初始化阶段，设置一些初始的参数
        cur_frame_idx = 0 #当前帧的索引，初始值为0
        projection_matrix = getProjectionMatrix2( #获取投影矩阵
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1) #将矩阵的第0和第1维进行转置，矩阵的第0维对应于行，第1维对应于列
        projection_matrix = projection_matrix.to(device=self.device) #将投影矩阵移动到指定的设备，所有的张量都必须在同一个设备上，才能进行计算
        # 创建两个CUDA事件，用于测量CUDA操作的时间
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            #检查gui向主进程传递消息的队列
            if self.q_vis2main.empty(): #如果为空
                if self.pause: #且处于暂停状态，则继续循环
                    continue
            else: #如果 q_vis2main 队列不为空
                data_vis2main = self.q_vis2main.get() #从队列中获取数据 data_vis2main
                self.pause = data_vis2main.flag_pause #根据这个数据更新暂停状态
                if self.pause: #如果更新后处于暂停状态
                    self.backend_queue.put(["pause"]) #向后端队列发送 "pause" 指令并继续循环
                    continue
                else:
                    self.backend_queue.put(["unpause"]) #否则，发送 "unpause" 指令

            # 检查前端消息队列
            if self.frontend_queue.empty(): #如果前端队列为空
                tic.record() #记录当前时间，用于计算处理时间
                if cur_frame_idx >= len(self.dataset): #检查当前帧索引是否大于等于数据集的长度，即是否已经处理完所有帧的数据
                    if self.save_results: #如果保存结果则进行误差评估和保存高斯分布
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init: #如果存在初始化请求，程序会暂停执行一段时间，并进行下一段循环
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0: #如果是单线程模式且请求了关键帧，同样会暂停执行一段时间并继续下一轮循环
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0: #如果未初始化且请求了关键帧，同样会暂停执行一段时间并继续下一轮循环
                    time.sleep(0.01)
                    continue

                #如果以上条件都不满足则：
                viewpoint = Camera.init_from_dataset( #从数据集中初始化相机视角对象，传入当前帧索引和投影矩阵
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config) #计算视角的梯度掩码

                self.cameras[cur_frame_idx] = viewpoint #将当前帧的视角信息存储在相机字典中

                if self.reset: #如果存在重置标志
                    self.initialize(cur_frame_idx, viewpoint) #初始化视图
                    self.current_window.append(cur_frame_idx) #将当前帧索引添加到当前窗口中
                    cur_frame_idx += 1 #然后继续下一帧的处理
                    continue

                self.initialized = self.initialized or ( #更新初始化状态，如果当前窗口大小等于指定窗口大小，则将初始化状态设置为 True
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint) #对当前帧进行跟踪，返回一个渲染数据包 render_pkg

                current_window_dict = {} #创建一个空字典，用于存储当前窗口的关键帧
                current_window_dict[self.current_window[0]] = self.current_window[1:] # 将当前窗口的关键帧存储到字典中，键为当前窗口的第一个帧，值为除第一个帧之外的其余帧
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window] # 据当前窗口的关键帧索引，获取对应的关键帧摄像机信息

                # 将高斯包装对象放入队列 q_main2vis 中，用于可视化。这个包装对象包含克隆的高斯模型、当前帧、关键帧列表和当前窗口的字典
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                # 如果有请求的关键帧
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx) #清理当前帧
                    cur_frame_idx += 1 #当前帧索引加一
                    continue #跳过当前循环，继续执行下一次循环

                last_keyframe_idx = self.current_window[0] #获取当前窗口的第一个关键帧索引
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval #计算当前帧索引与上一个关键帧索引之间的时间间隔是否大于等于设定的关键帧间隔
                curr_visibility = (render_pkg["n_touched"] > 0).long() #根据渲染数据中当前帧触及的点数是否大于0，判断当前帧的可见性，并将结果转换为长整型数据类型
                create_kf = self.is_keyframe( #判断是否需要创建新的关键帧
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility, #遮挡感知的可见性
                )
                # 如果当前窗口的帧数小于指定的窗口大小，则将当前帧添加到窗口中
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or( #计算当前帧可见性和上一个关键帧的可见性的并集中非零元素的数量
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and( #计算当前帧可见性和上一个关键帧的可见性的交集中非零元素的数量
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union #计算交集与并集的比值，表示当前帧在上一个关键帧的可见性范围内的点所占的比例
                    # 判断是否需要创建关键帧，条件是当前帧与上一个关键帧之间的时间间隔大于等于关键帧间隔，并且点的比例小于指定的阈值 kf_overlap
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )

                # 如果是单线程模式
                if self.single_thread:
                    create_kf = check_time and create_kf #进一步根据时间间隔和之前的判断结果确定是否需要创建新的关键帧
                # 如果需要创建关键帧
                if create_kf:
                    self.current_window, removed = self.add_to_window(#将当前帧添加到当前窗口中，并返回更新后的当前窗口和已移除的关键帧（如果有的话）
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:# 如果是单目摄像头且地图尚未初始化且已移除了关键帧
                        self.reset = True #将重置标志设置为True。因为如果地图尚未初始化且已移除了关键帧，那么就需要重置系统，也即需要重新初始化
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe( #根据渲染包的深度和不透明度信息添加新的关键帧
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe( #请求添加关键帧
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else: #如果不需要创建关键帧，那么就cleanup
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0 #检查当前关键帧索引列表的长度是否是保存轨迹关键帧间隔的整数倍
                ):
                    #ATE评估
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record() #记录当前时间
                torch.cuda.synchronize() #同步CUDA设备，确保前面的操作已经完成
                if create_kf: #如果需要创建关键帧
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc) #计算从记录开始到当前时间的持续时间 duration
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000)) #根据持续时间控制添加关键帧的频率，使其不超过3帧每秒的速率
            else:
                data = self.frontend_queue.get() #从前端队列中获取数据
                if data[0] == "sync_backend":
                    self.sync_backend(data) #将数据传递给后端进行同步

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
