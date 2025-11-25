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
import copy
import numpy as np
import os
import sys
import torch
import torchvision.transforms as transforms
import uuid
from PIL import ImageFilter
from argparse import ArgumentParser, Namespace
from fused_ssim import fused_ssim as fast_ssim
from random import randint
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from cuda_config import devices
from gaussian_renderer import render, network_gui_ws
from scene import Scene, GaussianModel
from utils.bayesian_losses import kl_divergence_loss, geometric_consistency_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from utils.prior_utils import likelihood_with_prior_update, likelihood
from utils.taming_utils import compute_gaussian_score

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from, websockets,
             score_coefficients,args):
    densify_iter_num = 0
    # 初始化训练迭代计数器，表示从第 0 次迭代开始
    # 如果从检查点恢复训练，first_iter会被设为保存的迭代次数
    first_iter = 0
    # 初始化 TensorBoard 日志记录器，用于保存训练过程中的指标（如损失、渲染质量）
    tb_writer = prepare_output_and_logger(dataset)
    # 创建 高斯模型（核心数据结构）
    # 球谐函数（Spherical Harmonics）的阶数，控制颜色和光照的表示能力
    # 优化器类型（如 "adam"、"sgd"）
    # 内部行为
    # 初始化高斯分布的参数（位置、颜色、透明度、协方差等）。
    # 根据optimizer_type配置优化器（如 Adam 的参数组）。
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # 构建 3D 场景，关联数据集和高斯模型
    # 关键操作：
    # 加载训练/测试相机位姿。
    # 初始化场景的边界框（bounding box）。
    # 可能执行高斯分布的初始空间分布（如从点云初始化）
    scene = Scene(dataset, gaussians)
    # 配置高斯模型的训练参数
    gaussians.training_setup(opt)

    # 设置渲染背景颜色和初始化CUDA事件计时器
    # 根据数据集配置选择白色或黑色背景，并转换为PyTorch张量
    # 在渲染时，未被3D对象覆盖的区域将填充此背景色（例如透明部分的底色）
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=devices.train_device())
    # CUDA事件计时器初始化 精确测量GPU代码的执行时间
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 获取训练相机视角并初始化索引 创建副本，避免直接修改原始数据
    viewpoint_stack = scene.getTrainCameras().copy()
    # 生成相机视角的索引列表 用于后续随机采样
    # 在训练时,可能需要随机选择不同视角的图片进行渲染和损失计算
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # todo // Edge-Aware-Score
    # 提取训练视角图像的边缘信息，并归一化存储
    all_edges = []
    for view in scene.getTrainCameras():
        edges_loss = get_edges(view.original_image).squeeze().cuda(device=devices.train_device())  # 提取边缘图
        edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))  # 归一化
        all_edges.append(edges_loss_norm.cpu())

    my_viewpoint_stack = scene.getTrainCameras().copy()  # 复制训练相机视角
    edges_stack = all_edges.copy()  # 复制边缘图数据
    # todo  // new
    counts_array = None
    # 初始化指数移动平均（EMA）损失记录
    # EMA损失用于稳定训练监控，避免瞬时波动干扰判断
    # 计算公式:ema_loss = α⋅current_loss + (1−α)⋅ema_loss 其中α是平滑系数（通常取 0.1~0.2）
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 动态选择渲染背景颜色
    # 若 opt.random_background=True：使用随机RGB颜色（torch.rand生成3个0~1之间的值）
    bg = torch.rand((3), device=devices.train_device()) if opt.random_background else background
    # perturbed_cam_dict = {}
    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render(cam, gaussians, pipe, background, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = memoryview(
                    (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        # 记录迭代开始时间（GPU计时） 与iter_end.record()配合，计算单次迭代耗时
        iter_start.record()
        if counts_array == None:
            # todo // 计算每一轮致密化迭代时的高斯点数量目标 num_steps （15000-500）// 100 = 145
            counts_array = get_count_array(len(scene.gaussians.get_xyz), args.budget, opt)
            # print(counts_array)
        # 更新学习率 根据当前迭代步 iteration动态调整学习率。
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代提升球谐函数（SH）阶数
        # 逐步增加球谐函数的active阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个训练相机视角
        # 从训练相机列表中随机选择一个视角用于当前迭代的渲染和损失计算
        # 如果 viewpoint_stack为空（所有相机已用完），重新加载所有训练相机并重置索引
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        # 取出对应相机，并从列表中移除（避免重复使用）
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        # 同步移除索引（保持一致性）
        _ = viewpoint_indices.pop(rand_idx)

        # 训练循环中的渲染部分,负责生成当前相机视角的渲染图像，并提取渲染过程中的关键信息
        # Render
        # 在指定迭代步（debug_from）启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # 执行渲染
        # viewpoint_cam：当前相机视角（位姿、内参等）
        # gaussians：3D高斯模型参数（位置、颜色、透明度等）
        # pipe：渲染管线配置（如分辨率、抗锯齿等）
        # bg：背景颜色
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss 计算损失
        # 计算渲染损失的核心部分，结合了L1损失（像素级绝对误差）和SSIM损失（结构相似性）来优化渲染质量
        # 从当前相机视角（viewpoint_cam）加载真实拍摄的图像，并确保其在GPU上（与渲染图像同设备）
        gt_image = viewpoint_cam.original_image.cuda(device=devices.train_device())  # 形状: [3, H, W]
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        survival_prob = gaussians.survival_prob
        post_survival_prob = likelihood(image, viewpoint_cam, viewspace_point_tensor, survival_prob.clone(),
                                        visibility_filter)
        # Bayesian losses
        # KL divergence loss
        # ps : KL损失通常需加权（如 0.1 * kl_loss）避免主导优化方向
        L_kl = kl_divergence_loss(post_survival_prob, survival_prob)
        # 组合混合损失
        # opt.lambda_dssim：SSIM损失的权重（通常为 0.2~0.5）
        # 1.0 - opt.lambda_dssim：L1损失的权重
        # 1.0 - ssim_value将相似性（[0,1]，越高越好）转为损失（越低越好）
        # L1主导（lambda_dssim小）：强调像素精度，可能忽略结构。
        # SSIM主导（lambda_dssim大）：保留边缘和纹理，但可能引入噪声
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + opt.lambda_kl * L_kl
        # + opt.lambda_geo * L_geo
        # Store loss components for logging
        loss_components = {
            "l1_loss": Ll1.item(),
            "ssim_loss": ssim_loss.item(),
            "kl_loss": L_kl.item(),
            "total_loss": loss.item()
        }

        loss.backward()

        iter_end.record()
        # 禁用梯度计算，确保内部的损失计算和EMA更新不会影响模型参数的梯度
        # EMA更新是纯数值操作，无需反向传播，节省计算资源
        with torch.no_grad():
            # Progress bar
            # 总损失EMA
            # EMA_loss=0.4×current_loss+0.6×EMA_loss_prev
            # 0.4：当前损失的权重（平滑因子），值越大对近期变化越敏感。
            # 0.6：历史EMA的权重，值越大曲线越平滑。
            # 过滤训练中损失的瞬时波动（如单次迭代的异常值）
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 每10次迭代更新进度条显示的损失值（减少刷新频率，提升性能）
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "N_GS": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            # 训练完成后清理进度条，避免输出混乱
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 训练监控、模型保存、点云动态优化（densification）、参数更新和检查点存储
            # 训练报告与监控
            # tb_writer：TensorBoard的SummaryWriter对象。
            # Ll1, loss, l1_loss：不同损失项（RGB L1、总损失、深度L1）。
            # iter_start.elapsed_time(iter_end)：单次迭代耗时（毫秒）
            training_report(tb_writer, iteration, Ll1, loss, loss_components, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            # todo Recovery-Aware Pruning (RAP)
            # RAP仅在早期阶段应用，因此它不会影响最终的高斯数量或训练速度。
            if iteration == 300:
                gaussians.only_prune(0.02)

            # Densification
            if opt.densify_from_iter < iteration < opt.densify_until_iter:
                # 收集优化统计量
                # 累积高斯点的梯度统计量（如位置梯度均值），指导后续分裂或剪枝
                gaussians.add_densification_stats_abs(viewspace_point_tensor, visibility_filter)
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration % opt.densification_interval == 0:
                    # cams: 每次迭代渲染的视角数量，-1表示使用所有训练视角
                    num_cams = args.cams
                    # iteration: 3400/6400
                    if args.cams == -1:
                        # 所有训练视角
                        num_cams = len(scene.getTrainCameras().copy())
                    edge_losses = []
                    camlist = []
                    for _ in range(num_cams):
                        if not my_viewpoint_stack:
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            edges_stack = all_edges.copy()
                        # 每次从栈顶取出一个视角和对应的边缘图
                        camlist.append(my_viewpoint_stack.pop())
                        edge_losses.append(edges_stack.pop())
                    # todo Edge-Aware-Score (EAS)
                    # 计算每个高斯点的重要性分数
                    gaussian_importance = compute_edge_score(camlist, edge_losses, gaussians, pipe, bg)
                    # gaussian_importance = compute_gaussian_score(scene, camlist, edge_losses, gaussians, pipe, bg,
                    #                                              score_coefficients, opt)

                    gaussians.densify_and_prune_GEO_Improved(gaussian_importance, 0.005,
                                                             counts_array[densify_iter_num + 1],
                                                             opt, iteration, scene.cameras_extent, densify_iter_num,
                                                             post_survival_prob)
                    # gaussians.densify_and_prune_Improved(gaussian_importance, 0.005,
                    #                                          counts_array[densify_iter_num + 1],
                    #                                          opt, iteration, scene.cameras_extent, densify_iter_num,
                    #                                          post_survival_prob)
                    densify_iter_num += 1
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity(0.03)
                # todo Recovery-Aware Pruning (RAP) 替换 初始的每隔3000步重置不透明度
                if iteration % opt.opacity_reset_interval == 300:
                    gaussians.only_prune(0.2, True)

            # Optimizer step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    # todo Muti-view Update -> MU 迭代15000次后，开始减少优化频率 15000-22500每5步优化一次，之后每20步优化一次
                    if iteration <= 15000:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.shoptimizer.step()
                        gaussians.shoptimizer.zero_grad(set_to_none=True)
                    elif iteration <= 22500:
                        if iteration % 5 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                    else:
                        if iteration % 25 == 0:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none=True)
                            gaussians.shoptimizer.step()
                            gaussians.shoptimizer.zero_grad(set_to_none=True)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)


# 从输入图像中提取边缘信息，其核心逻辑是通过PIL库的简单边缘检测滤波器实现
def get_edges(image):
    image_pil = transforms.ToPILImage()(image)  # 转为PIL图像
    image_gray = image_pil.convert('L')  # 转为灰度图
    # mageFilter.FIND_EDGES是PIL内置的简单边缘检测算子，本质是一个高通滤波器（近似Sobel或Prewitt算子）
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)  # 边缘检测
    image_edges_tensor = transforms.ToTensor()(image_edges)  # 转回Tensor
    return image_edges_tensor


"""
对输入张量进行归一化处理，其核心逻辑是将非零有效值除以均值，并处理无效值（NaN和零值）
"""


def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0  # 将NaN置零
    valid_indices = (value_tensor > 0)  # 标记有效值（>0）
    valid_value = value_tensor[valid_indices].to(torch.float32)  # 提取有效值
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)  # 初始化输出
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)  # 归一化

    return ret_value


"""
计算高斯点边缘重要性分数（Edge Score）的核心函数
其目的是通过多视角渲染和边缘损失加权，评估每个高斯点在边缘区域的贡献强度
"""


def compute_edge_score(camlist, edge_losses, gaussians, pipe, bg):
    # 获取高斯点数量
    num_points = len(gaussians.get_xyz)
    # 初始化每个高斯点的重要性分数为0
    gaussian_importance = torch.zeros(num_points, device=devices.train_device(),
                                      dtype=torch.float32)
    # 初始化一个布尔数组，标记至少在一个视角中可见的高斯点
    visibility_filter_all = torch.zeros(num_points, device=devices.train_device(), dtype=bool)
    # 多视角渲染与加权累积
    for view in range(len(camlist)):
        # 渲染当前视角图像，并获取渲染过程中的关键信息
        my_viewpoint_cam = camlist[view]
        # 获取当前视角的边缘损失图，并确保其在GPU上
        pixel_weights = edge_losses[view].cuda(devices.train_device())
        # 渲染当前视角
        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg, pixel_weights=pixel_weights)
        # 归一化累积权重，确保不同视角的贡献可比较
        loss_accum = normalize(render_pkg["accum_weights"])
        # 获取当前视角下可见的高斯点掩码
        visibility_filter = render_pkg["visibility_filter"].detach()
        # 仅对可见的高斯点进行加权累积
        gaussian_importance[visibility_filter] += loss_accum[visibility_filter] / len(camlist)
        visibility_filter_all[visibility_filter] = True

    # 结果修正，确保仅对至少在一个视角中可见的点返回分数（隐式将不可见点分数置零）
    gaussian_importance[visibility_filter_all] = gaussian_importance[visibility_filter_all]
    return gaussian_importance


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


"""
3D高斯泼溅（3D Gaussian Splatting）训练过程中的监控与报告模块，
主要负责记录训练指标、可视化渲染结果，并在特定迭代步评估模型性能
"""


def training_report(tb_writer, iteration, Ll1, loss, loss_components, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc,
                    renderArgs):
    """
    训练过程中的报告函数，负责记录损失值到TensorBoard
    特别关注KL损失等各个损失组件的可视化
    
    :param tb_writer: TensorBoard的SummaryWriter对象
    :param iteration: 当前训练迭代步数
    :param Ll1: L1损失值
    :param loss: 总损失值
    :param loss_components: 损失组件字典，包含各种损失项（如KL损失）
    :param l1_loss: L1损失计算函数
    :param elapsed: 单次迭代耗时
    :param testing_iterations: 需执行测试的迭代步列表
    :param scene: 场景对象
    :param renderFunc: 渲染函数
    :param renderArgs: 渲染参数
    """
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # 记录所有损失组件，特别是KL损失
        # loss_components = {
        #     "l1_loss": Ll1.item(),
        #     "ssim_loss": ssim_loss.item(),
        #     "kl_loss": L_kl.item(),
        #     "total_loss": loss.item()
        # }
        if loss_components is not None:
            for key, value in loss_components.items():
                tb_writer.add_scalar(f'train_loss_patches/{key}', value, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},  # 全部测试相机（scene.getTestCameras()）
                              {'name': 'train', 'cameras': scene.getTrainCameras()})  # 全部训练相机（scene.getTestCameras()）

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染图像并与GT比较
                    # 渲染当前视角图像（renderFunc），值域钳制到 [0, 1]。
                    # 加载真实图像（GT）并确保在GPU上。
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device=devices.train_device()), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        # 前5个视角的渲染结果
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        # 仅在第一次测试时记录GT（避免冗余）
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    # 计算指标
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # 输出结果
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def get_count_array(start_count, multiplier, opt):
    budget = multiplier
    # 计算总步数 num_steps （15000-500）// 100 = 145
    num_steps = ((opt.densify_until_iter - opt.densify_from_iter) // opt.densification_interval) -1
    # 计算增长的最低斜率
    slope_lower_bound = (budget - start_count) / num_steps
    # 计算二次增长参数
    k = 2 * slope_lower_bound

    a = (budget - start_count - k * num_steps) / (num_steps * num_steps)
    b = k
    c = start_count
    # count(x) = a * x² + b * x + c
    # x：当前步数（0 ≤ x < num_steps）。
    # a、b、c：通过边界条件计算得出，确保：
    # count(0) = start_count
    # count(num_steps) = budget
    # 初始斜率 k为 2 * slope_lower_bound（控制增长速率）
    values = [int(1 * a * (x ** 2) + (b * x) + c) for x in range(num_steps + 1)]  # 生成数量序列

    return values


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[12000 + 3000 * (i + 1) for i in range(6)])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cams", type=int, default=10)
    parser.add_argument("--websockets", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if (args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    score_coefficients = {
        'view_importance': 50,  # 视角一致性的权重
        'edge_importance': 50,  # 边缘锐度的权重
        'mse_importance': 50,  # 均方误差的权重
        'grad_importance': 25,  # 梯度平滑性的权重
        'dist_importance': 50,  # 点云分布均匀性的权重
        'opac_importance': 100,  # 不透明度控制的权重
        'dept_importance': 5,  # 深度一致性的权重
        'loss_importance': 10,  # 总损失的权重
        'radii_importance': 10,  # 高斯半径的权重
        'scale_importance': 25,  # 尺度一致性的权重
        'count_importance': 0.1,  # 点云数量的权重
        'blend_importance': 50  # 混合效果的权重
    }

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.debug_from,
        args.websockets,
        score_coefficients,
        args
    )

    # All done
    print("\nTraining complete.")
