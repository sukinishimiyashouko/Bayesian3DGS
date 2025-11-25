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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from cuda_config import devices


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, pixel_weights=None, bayesian_rendering=False,
           sample_rendering=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 初始化屏幕空间坐标和设置光栅化参数
    # 在光栅化过程中，3D高斯点会投影到2D屏幕空间，此张量将记录投影坐标，以便计算几何参数的梯度（如点云位置 xyz的梯度）
    # 创建一个与3D高斯点云位置（pc.get_xyz）形状相同的张量，用于存储2D屏幕空间坐标
    # Improved-gs专用
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True,
                                     device=devices.train_device()) + 0
    # Taming-gs 专用
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=devices.train_device()) + 0
    try:
        # retain_grad():确保中间变量screenspace_points的梯度在反向传播时保留
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 计算相机水平和垂直视场角（FoV）的正切值
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 配置光栅化参数并准备高斯点数据，最终通过光栅化器生成图像
    # 配置光栅化参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        # 是否预过滤高斯点
        prefiltered=False,
        debug=pipe.debug,
        pixel_weights=pixel_weights
    )
    # 初始化光栅化器
    # 基于CUDA实现高效并行光栅化，支持可微分渲染
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # 准备高斯点数据
    means3D = pc.get_xyz  # 高斯点的3D位置 [N, 3]
    means2D = screenspace_points  # 高斯点的2D屏幕坐标（初始为零，光栅化器内计算）
    opacity = pc.get_opacity  # 高斯点的不透明度 [N, 1]
    survival_prob = None
    if bayesian_rendering and hasattr(pc, '_survival_logit') and pc._survival_logit.shape[0] > 0:
        survival_prob = pc.survival_prob_activation(pc._survival_logit)
        if sample_rendering:
            with torch.no_grad():
                z = torch.bernoulli(survival_prob).to(devices.train_device())
            # Apply the sampled mask
            opacity = opacity * z
        else:
            opacity = opacity * survival_prob

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    # 协方差矩阵计算策略
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling  # 高斯点的尺度 [N, 3]
        rotations = pc.get_rotation  # 高斯点的旋转（四元数） [N, 4]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 处理高斯点的颜色属性（球谐系数或预计算颜色）并调用光栅化器生成图像
    # 初始化球谐系数（shs）和预计算颜色（colors_precomp）为 None，后续根据条件赋值
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # Python计算球谐系数转RGB
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 使用光栅化器内部计算球谐颜色
            # 分离球谐系数（separate_sh=True）
            if pipe.separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Apply Bayesian rendering to precomputed colors as well
    if bayesian_rendering and hasattr(pc, '_survival_logit') and pc._survival_logit.shape[
        0] > 0 and colors_precomp is not None:
        survival_prob = pc.survival_prob_activation(pc._survival_logit)
        if sample_rendering:
            with torch.no_grad():
                z = torch.bernoulli(survival_prob).to(devices.train_device())
            colors_precomp = colors_precomp * z.repeat(1, 3)
        else:
            colors_precomp = colors_precomp * survival_prob.repeat(1, 3)

    # 调用光栅化器生成图像
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if pipe.separate_sh:
        rendered_image, radii, counts, lists, listsRender, listsDistance, centers, depths, my_radii, accum_weights, accum_count, accum_blend, accum_dist = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,  # 主色（漫反射）
            shs=shs,  # 高阶球谐（镜面反射）
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        # 染的RGB图像、高斯点的屏幕空间半径和深度图
        # rendered_image:[3, height, width]（3通道RGB，值范围通常为 [0, 1]或 [0, 255]）
        # radii: [N],每个3D高斯点投影到2D屏幕后的近似像素半径，反映该点在图像中的覆盖范围
        # depth_image:[1, height, width],与RGB图像对应的深度信息，表示每个像素到相机的距离（或Z缓冲值）
        rendered_image, radii, counts, lists, listsRender, listsDistance, centers, depths, my_radii, accum_weights, accum_count, accum_blend, accum_dist = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            # cov3D_precomp=None 触发内部计算
            cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "counts": counts,
        "lists": lists,
        "listsRender": listsRender,
        "listsDistance": listsDistance,
        "gaussian_centers": centers,
        "gaussian_depths": depths,
        "gaussian_radii": my_radii,
        "accum_weights": accum_weights,
        "accum_count": accum_count,
        "accum_blend": accum_blend,
        "accum_dist": accum_dist
    }
