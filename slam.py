import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True) #创建了一个CUDA事件对象 start，用于记录开始时间
        end = torch.cuda.Event(enable_timing=True) #创建了一个CUDA事件对象 end，用于记录结束时间

        start.record() #记录开始时间

        self.config = config #获取配置文件
        self.save_dir = save_dir #获取保存结果的目录路径

        # 通过munchify 函数将 config["model_params"] 这个字典转换为了一个 munch 对象
        # munch库提供了可以像调用对象属性一样使用字典的方法（dic.key）
        # 将 config 中的参数转换为 model_params,opt_params 和 pipeline_params 三个对象，并保存在类属性中
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # sh系数
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # 初始化高斯模型（并设置一些列参数）
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0) #初始化高斯模型的学习率（空间学习率）
        # 加载数据集
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        # 设置高斯模型的优化参数
        self.gaussians.training_setup(opt_params)
        # 设置背景颜色
        bg_color = [0, 0, 0] #黑色
        # 创建了一个 PyTorch 张量来表示背景颜色，并将其存储在类的属性中。这个张量会被放置在 CUDA 设备上
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 创建了两个多进程队列，用于前端和后端之间的通信
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        # 根据是否使用 GUI，选择使用 mp.Queue() 或者 FakeQueue() 创建一个模拟的队列对象
        # 这样做的目的是根据程序的运行模式来选择不同类型的队列。如果程序需要与 GUI 进程进行通信，就使用真实的多进程队列；
        # 如果不需要，就使用模拟的队列，以避免在没有 GUI 的情况下引发异常。
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # 创建了前端和后端对象，并设置属性
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        # 创建了一个用于显示参数的 GUI 对象
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        # 创建了一个后台进程 backend_process，并开始执行后端的运行函数 self.backend.run()
        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui: #如果使用 GUI，创建一个 GUI 进程 gui_process 并开始执行
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start() #启动 GUI 进程，让它开始执行 GUI 的运行函数
            time.sleep(5) #等待5秒

        backend_process.start() #启动后台进程，让它开始执行后端的运行函数

        # 启动前端的运行函数
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record() #记录结束时间
        torch.cuda.synchronize() #同步点，它确保在继续执行之前，所有之前提交的 CUDA 操作都已经完成
        # empty the frontend queue
        N_frames = len(self.frontend.cameras) #获取了前端处理的相机帧数量，用于计算帧率
        FPS = N_frames / (start.elapsed_time(end) * 0.001) #计算了帧率，单位是帧/秒
        # 打印到终端查看
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians #前端处理后的高斯模型参数
            kf_indices = self.frontend.kf_indices #关键帧索引

            # 计算 Absolute Trajectory Error (ATE)绝对轨迹误差。这个函数接受相机位置、关键帧索引等参数，SLAM中计算系统估计的路径与真实路径之间的差异
            ATE = eval_ate(
                self.frontend.cameras, #SLAM系统前端处理过程中收集的相机或传感器数据，含有位姿
                self.frontend.kf_indices, #关键帧索引。在SLAM中，关键帧是指那些包含足够信息，能够用于优化地图和路径的特定帧
                self.save_dir, #结果保存的目录
                0,
                final=True,
                monocular=self.monocular,
            )

            # 执行渲染评估。这个函数接受相机位置、高斯模型参数、数据集等参数，计算出渲染结果
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            # 创建一个表格对象，用于记录评估结果
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data( #将渲染评估结果添加到表格中
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty(): #清空前端队列中的所有消息
                frontend_queue.get()
            backend_queue.put(["color_refinement"]) #消息（["color_refinement"]）放入后端队列
            while True: #在一个无限循环中等待从前端队列中获取同步信号，并从后端队列中获取最新的高斯模型参数
                if frontend_queue.empty(): #如果前端队列为空，就让线程休眠0.01秒，然后继续下一次循环
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get() #从前端队列中获取消息
                if data[0] == "sync_backend" and frontend_queue.empty(): #用于检查获取到的消息是否是"sync_backend"，并且前端队列是否为空
                    gaussians = data[1] #从消息中获取高斯数据
                    self.gaussians = gaussians
                    break
            #评估渲染结果的质量，包括计算图像的峰值信噪比PSNR，结构相似性指数SSIM和感知图像差异LPIPS
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data( #将渲染结果的评估指标添加到指标表
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"]) #向后端队列中放入一个"stop"消息，通知后台进程停止运行
        backend_process.join() #等待后台进程结束
        Log("Backend stopped and joined the main thread")
        if self.use_gui: #如果使用GUI，就向主线程到可视化线程的队列（q_main2vis）中放入一个包含"finish=True"的高斯数据包
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join() #等待GUI进程结束
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
