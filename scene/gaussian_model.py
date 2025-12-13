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
import faiss
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from cuda_config import devices
from utils.prior_utils import compute_color_based_prior, compute_scale_based_prior, combine_priors

from sklearn.neighbors import KDTree

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class GaussianModel:

    def __init__(self, sh_degree, optimizer_type="default"):
        # 当前激活的球谐函数阶数（初始为0，可能逐步增加）
        self.active_sh_degree = 0
        # 优化器类型（如 "adam"），默认为 "default"
        self.optimizer_type = optimizer_type
        # 球谐函数（Spherical Harmonics）的最大阶数，控制颜色和光照的表示能力
        self.max_sh_degree = sh_degree
        """
        高斯模型参数：初始化为空张量（torch.empty(0)），后续根据输入数据动态填充
        """
        # 高斯分布的中心位置（3D坐标）
        self._xyz = torch.empty(0)
        # 球谐函数的第0阶（直流分量）颜色特征
        self._features_dc = torch.empty(0)
        # 球谐函数的高阶（≥1阶）颜色特征
        self._features_rest = torch.empty(0)
        # 高斯分布的缩放因子（3D椭球的形状）
        self._scaling = torch.empty(0)
        # 高斯分布的旋转（四元数或旋转矩阵）
        self._rotation = torch.empty(0)
        # 高斯分布的不透明度（控制渲染时的可见性）
        self._opacity = torch.empty(0)
        # Bayesian framework parameter
        self._survival_logit = torch.empty(0)  # Bernoulli parameter for survival probability
        """
        训练状态变量
        """
        # 位置梯度的累积量（可能用于稀疏化控制）
        self.xyz_gradient_accum = torch.empty(0)
        # 分母项（可能用于梯度归一化或优化计算）
        self.denom = torch.empty(0)
        # 优化器实例
        self.optimizer = None
        # todo // SH feature optimizer
        self.shoptimizer = None
        # 控制高斯分布的密度（稀疏化/致密化）
        self.percent_dense = 0
        # 空间学习率缩放因子（可能用于自适应调整学习率）
        self.spatial_lr_scale = 0
        """
        构建协方差矩阵
        输入:
        scaling: 高斯分布的缩放因子（3D椭球的形状）。
        scaling_modifier: 缩放修正系数（可能用于动态调整）。
        rotation: 高斯分布的旋转（四元数或旋转矩阵）
        :return:对称的协方差矩阵（描述高斯分布的形状和方向）
        将可学习的缩放和旋转参数转换为物理可解释的协方差矩阵。
        """

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        """缩放参数的激活函数"""
        # 缩放参数的实际值 = exp(学习值)
        self.scaling_activation = torch.exp
        # 学习值 = log(实际值)
        self.scaling_inverse_activation = torch.log
        """协方差矩阵的激活函数"""
        # 将缩放和旋转参数转换为协方差矩阵（通过上述定义的函数）
        self.covariance_activation = build_covariance_from_scaling_rotation
        """不透明度（Opacity）的激活函数"""
        # 不透明度 ∈ (0, 1)
        self.opacity_activation = torch.sigmoid
        # 逆函数（从实际值反推学习值）
        self.inverse_opacity_activation = inverse_sigmoid
        """旋转参数的激活函数"""
        # 对旋转参数（如四元数）进行归一化，保持单位长度（避免数值不稳定）
        self.rotation_activation = torch.nn.functional.normalize
        # Bayesian framework activation functions
        self.survival_prob_activation = torch.sigmoid
        self.inverse_survival_prob_activation = inverse_sigmoid

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._survival_logit,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self._survival_logit,
         xyz_gradient_accum,
         denom,
         opt_dict,
         shopt_dict,
         self.spatial_lr_scale) = model_args

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def survival_prob(self):
        """生存概率，通过sigmoid函数将logit转换为概率值"""
        if self._survival_logit.numel() == 0:
            return torch.ones_like(self.get_opacity[:0]) if self._opacity.numel() > 0 else torch.empty(0)
        return self.survival_prob_activation(self._survival_logit)

    def survival_entropy(self, p):
        """计算生存概率的熵，衡量不确定性"""
        # 处理p为None或空的情况
        if p is None or p.numel() == 0:
            return torch.tensor(0.0, device=devices.train_device())
        return -p * torch.log(p + 1e-10) - (1 - p) * torch.log(1 - p + 1e-10)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    """
    从点云（PCD）初始化高斯分布的核心逻辑
    pcd：COLMAP生成的稀疏点云（BasicPointCloud对象，包含 points和 colors）。
    cam_infos：相机参数列表（用于计算高斯初始协方差）。
    spatial_lr_scale：空间学习率缩放因子（控制参数更新速度）
    """

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        # 初始化高斯点云 完成点云数据的加载、颜色转换和特征初始化
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda(device=devices.train_device())
        # 颜色转换 (RGB → SH)
        # 将RGB颜色转换为球谐函数的零阶系数（SH基的DC分量）
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda(device=devices.train_device()))
        # 球谐特征初始化 为每个点初始化球谐系数张量，形状为 (N, 3, (max_sh_degree+1)^2) 例如max_sh_degree=3 最终=16
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda(
            device=devices.train_device())
        features[:, :3, 0] = fused_color  # 零阶系数（颜色）
        # features[:, 3:, 1:] = 0.0  # 高阶系数初始化为零,改不改都无所谓,已经初始化过了
        features[:, 3:, 1:] = 0.0  # 高阶系数初始化为零
        # 例如:Number of points at initialisation :  80861 [20/09 14:03:02]
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 计算每个点到其最近邻点的距离的平方
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda(device=devices.train_device())), 0.0000001)
        # torch.sqrt计算距离平方的平方根，得到实际距离 torch.log取对数，使尺度参数在更合理的范围内 并增加维度
        # repeat(1, 3)将结果复制到3个维度(x,y,z)即复制三次，使每个点有3个相同的尺度值
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # 创建旋转四元数数组，形状为(点数, 4)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=devices.train_device())
        # 将四元数的第一个分量(w)设为1，其余(x,y,z)保持0 这表示初始状态没有旋转
        rots[:, 0] = 1
        # 处理3D高斯分布的点云数据，主要用于初始化一些参数
        # distCUDA2计算点云中每个点到其最近邻点的距离平方
        # torch.clamp_min确保所有距离值不小于一个极小值(0.0000001)，防止出现零或负值
        # 结果是每个点对应的最小距离平方值
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=devices.train_device()))

        # todo new //
        # 1.计算颜色先验分量
        color_prior = compute_color_based_prior(fused_color)
        # 2.计算尺度先验分量
        scale_prior = compute_scale_based_prior(scales)
        # 3.计算几何位置先验分量
        # geometry_prior = compute_approximate_density(fused_point_cloud)
        # 4.组合鲜艳分量得到初始生存概率
        initial_survival_probs = combine_priors(color_prior, scale_prior)
        # 5. 将概率转换为logit形式，便于优化
        # 消融实验,初始化生存概率为0.9
        # survival_logit = self.inverse_survival_prob_activation(
        #     0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=devices.train_device()))
        survival_logit = self.inverse_survival_prob_activation(initial_survival_probs)
        # 初始化3D高斯泼溅（3D Gaussian Splatting）的可学习参数，主要包括点云位置、颜色、尺度、旋转、不透明度等，
        # 并可能涉及曝光参数（用于HDR或NeRF类渲染）
        # 定义点云的位置（3D坐标）为可学习参数 (N,3)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 定义颜色的主成分（DC分量，通常是漫反射颜色）为可学习参数
        # features ： (N,3,16)
        # .transpose(1, 2)：调整维度顺序 -> (N,1,3)
        # .contiguous()：确保内存连续，提升计算效率
        # 优化每个点的基色（如RGB值）
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # 定义颜色的高阶球谐系数（非DC分量，如镜面反射、光照变化）为可学习参数
        # 同上 -> (N,15,3) 优化光照的高频细节（如阴影、高光）
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 定义每个高斯分布的尺度（例如:协方差矩阵的对角线）为可学习参数
        # 优化高斯分布的“大小”，控制渲染时的模糊程度
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # 定义每个高斯分布的旋转（四元数表示）为可学习参数
        # 优化高斯分布的方向（如拉伸或倾斜）
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # 定义每个高斯分布的不透明度为可学习参数 控制点的可见性（透明/不透明）(N,1)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 贝叶斯框架下的生存概率参数
        self._survival_logit = nn.Parameter(survival_logit.requires_grad_(True))

    # 初始化优化器参数和设置梯度累积变量
    def training_setup(self, training_args):
        # 在训练过程中动态调整点云密度（例如控制高斯分布的稀疏性）
        self.percent_dense = training_args.percent_dense
        # 初始化两个缓冲区，用于梯度累积和归一化分母
        # xyz_gradient_accum：累积点位置（_xyz）的梯度，形状为 (N, 1)（N是点数）
        # 在训练过程中，梯度会被累加到 xyz_gradient_accum，可能用于点云的动态调整（如新增或删除点）
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())
        # denom：可能用于梯度归一化的分母项（如Adam优化器的二阶动量）
        # denom可能用于梯度裁剪或自适应学习率调整。
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())
        # 定义不同参数的学习率分组，用于优化器（如Adam）
        # params：要优化的参数列表（如 self._xyz）。
        # lr：各参数组的学习率。
        # name：参数组名称（用于调试或日志记录）
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._survival_logit], 'lr': training_args.survival_lr_init, "name": "survival_logit"}
        ]
        # todo // SH feature lr is 1/20 of the rest features lr
        sh_l = [{'params': [self._features_rest], 'lr': training_args.shfeature_lr / 20.0, "name": "f_rest"}]

        # 根据配置选择优化器类型
        # l：之前定义的参数组列表（包含不同参数的学习率）。
        # lr=0.0：全局学习率设为0，因为参数组中已单独指定学习率。
        # eps=1e-15：极小的数值稳定性常数（防止除以零）。
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.survival_scheduler_args = get_expon_lr_func(lr_init=training_args.survival_lr_init,
                                                         lr_final=training_args.survival_lr_final,
                                                         lr_delay_steps=500,
                                                         lr_delay_mult=training_args.survival_lr_delay_mult,
                                                         max_steps=training_args.densify_until_iter)

    """
    训练过程中对不同参数组的学习率进行独立调整
    根据当前训练迭代步数iteration,更新不同参数组的学习率
    """

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # 正常的学习率调度
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "survival_logit":
                lr = self.survival_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def construct_list_of_attributes_with_survival(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add survival logit for Bayesian framework
        l.append('survival_logit')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 构建属性列表，根据_survival_logit是否存在决定是否包含
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 构建属性数组，只包含存在的属性
        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        # 连接所有属性
        attributes = np.concatenate(attributes_list, axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_with_survival(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 构建属性列表，根据_survival_logit是否存在决定是否包含
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_with_survival()]

        # 构建属性数组，只包含存在的属性
        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        # 只有当_survival_logit不为None时，才添加到属性列表
        if hasattr(self, '_survival_logit') and self._survival_logit is not None:
            # Get survival probability (logit space) for Bayesian framework
            survival_logit = self._survival_logit.detach().cpu().numpy()
            attributes_list.append(survival_logit)

        # 连接所有属性
        attributes = np.concatenate(attributes_list, axis=1)

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    """
    重置所有高斯点不透明度（opacity）的具体实现，目的是将不透明度重置为一个较低的值，同时保持可优化性。
    """

    def reset_opacity(self, min_opacity):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * min_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load survival probability if present in the PLY file, otherwise set to None
        if "survival_logit" in plydata.elements[0].properties:
            # If already in logit space
            survival_logit = np.asarray(plydata.elements[0]["survival_logit"])[..., np.newaxis]
        else:
            # Set to None instead of initializing with default value
            survival_logit = None

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=devices.train_device()).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=devices.train_device()).transpose(1,
                                                                                                  2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=devices.train_device()).transpose(1,
                                                                                                     2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=devices.train_device()).requires_grad_(True))
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device=devices.train_device()).requires_grad_(True))
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device=devices.train_device()).requires_grad_(True))
        # Set survival probability parameter for Bayesian framework (logit space)
        if survival_logit is not None:
            if isinstance(survival_logit, np.ndarray):
                survival_logit = torch.tensor(survival_logit, dtype=torch.float, device=devices.train_device())
            self._survival_logit = nn.Parameter(survival_logit.requires_grad_(True))
        else:
            self._survival_logit = None
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    """
    根据掩码（mask）删除无效的高斯点，即父点，并同步更新模型参数和优化器状态
    """

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._survival_logit = optimizable_tensors["survival_logit"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

    """
    用于将新点（克隆或分裂生成）的参数合并到优化器（如Adam）的核心函数
    """

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_survival_logit):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "survival_logit": new_survival_logit
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._survival_logit = optimizable_tensors["survival_logit"]
        # 重置梯度累积器 为新点初始化梯度累积缓冲区（xyz_gradient_accum）和计数器（denom），后续训练中会重新累积
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())

    # 剪枝（Pruning）低不透明度高斯点的方法，支持绝对阈值和分位数比例两种剪枝模式
    def only_prune(self, min_opacity, percent=False):
        if percent is True:
            # 根据给定的百分比剪枝最低不透明度的点
            opacity_array = self.get_opacity.detach().flatten()  # 展平不透明度张量
            q = min_opacity  # 分位数比例（如0.02）
            min_opacity = torch.quantile(opacity_array, q)  # 计算实际阈值
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        else:
            # 直接使用给定的最小不透明度阈值进行剪枝
            prune_mask = (self.get_opacity < min_opacity).squeeze()

        valid_points_mask = ~prune_mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._survival_logit = optimizable_tensors["survival_logit"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=devices.train_device())

        torch.cuda.empty_cache()

    # 动态调整点云密度（Densification）和剪枝（Pruning）的核心逻辑，
    # 通过分析梯度、透明度和几何尺寸，决定哪些高斯点需要分裂（增加细节）或删除（减少冗余）
    def densify_and_prune_Improved(self, scores, min_opacity, budget, opt, iteration, extent, iter_num,
                                   post_survival_prob):
        # 计算平均梯度 即计算每个高斯点的平均屏幕空间梯度幅值（累积梯度除以更新次数）
        grad_vars = self.xyz_gradient_accum / self.denom
        # 将无效值（如除零）置为0，避免后续操作崩溃
        grad_vars[grad_vars.isnan()] = 0.0
        # 默认梯度阈值，用于决定是否增加高斯分布
        min_grad = opt.densify_grad_threshold
        # todo // Ours
        # Compute survival probability (for Bayesian framework)
        # survival_prob = self.survival_prob
        entropy = self.survival_entropy(post_survival_prob).squeeze(-1)  # 形状 [N]
        # Modify densification criteria with Bayesian uncertainty
        # Create modified gradient norm that combines gradient and uncertainty
        # 计算梯度范数 梯度范数越大，表示该点在训练中需要更多优化（如位于边缘或高变化区域）
        grad_norm = torch.linalg.norm(grad_vars, dim=-1)  # 形状 [N]
        # 联合梯度与熵的修正度量
        if iteration <= 7500:
            modified_grad = grad_norm * (1 + 2 * entropy)
        elif iteration <= 13500:
            modified_grad = grad_norm * (1 + 1.5 * entropy)
        else:
            # modified_grad = grad_norm * survival_prob.squeeze(-1)
            modified_grad = grad_norm * (1 + entropy)
        # 动态控制高斯点分裂（Splitting）和剪枝（Pruning）的核心逻辑
        # 主要功能是根据梯度幅值和当前点云状态，调整分裂预算（budget）和最小梯度阈值（min_grad）
        if iteration > 13500:
            min_grad = min_grad / 1.5  # 放宽梯度阈值

        # 梯度合格点筛选
        # torch.norm(grad_vars, dim=-1)：计算每个点的梯度L2范数（形状 [N]）
        # torch.where(condition, True, False)：生成布尔掩码，标记梯度幅值 >= min_grad的点
        # grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)
        # grad_qualifiers = torch.where(modified_grad >= min_grad, True, False)
        grad_qualifiers = torch.where(modified_grad >= min_grad, True, False)
        clone_qualifiers = (torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent)
        split_qualifiers = (torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent)
        # clone_qualifiers = torch.where(entropy >= 0.45, True, False)
        # split_qualifiers = torch.logical_and(torch.where(entropy >= 0.15, True, False),torch.where(entropy < 0.45, True, False))
        # 克隆与分裂条件
        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers)

        total_clones = torch.sum(all_clones).item()
        total_splits = torch.sum(all_splits).item()
        # 计算当前点数和预算的关系，确保不超过预算
        # total_sum = torch.sum(grad_qualifiers).item()  # 合格点数
        # 获取当前所有高斯点
        curr_points = len(self.get_xyz)
        # 动态分配克隆（clone）和分裂（split）操作的预算，确保在增加高斯点数量时不超过总点数限制（budget），同时合理分配克隆和分裂的配额
        budget = min(budget, total_clones + total_splits + curr_points)
        # 克隆与分裂预算分配 克隆和分裂的预算与其候选点数成正比
        clone_budget = ((budget - curr_points) * total_clones) // (total_clones + total_splits)
        split_budget = budget - curr_points - clone_budget
        self.densify_and_clone_bayesian(scores.clone(), clone_budget, all_clones)
        # 动态控制高斯点分裂（Splitting）和剪枝（Pruning）的核心逻辑
        # 主要功能是根据预算和当前点云状态，执行高斯点分裂和剪枝操作
        # todo Long axis splitting of Gaussians - LAS  长轴分裂
        self.long_axis_split(scores.clone(), split_budget, all_splits, opt.split_distance, opt.opacity_reduction)
        # self.densify_and_split_taming(scores.clone(),split_budget,all_splits)
        # 删除透明度低于 min_opacity的点（如 opacity < 0.005），这些点对渲染贡献极小
        opacity_mask = (self.get_opacity < min_opacity).squeeze()
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # survival_prob_down = (self.survival_prob < survival_prob_threshold).squeeze()
        # prune_mask = torch.logical_or(torch.logical_or(opacity_mask, big_points_ws), survival_prob_down)
        prune_mask = opacity_mask
        # if iteration < 14500:
        #     self.prune_points(prune_mask)
        remove_budget = int(0.5 * torch.sum(prune_mask).item())
        if remove_budget > 0 and iter_num < 27:
            entropy = self.survival_entropy(self.survival_prob).squeeze(-1)
            entropy_mask = entropy <= 0.05
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            # 将重要性分数转换为剪枝概率
            # 1 / (scores + 1e-6)：分数越低 → 概率越高（更可能被剪枝）
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores)
            # 基于重要性的采样
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device=devices.train_device())
            # torch.multinomial:按概率分布（padded_importance）随机采样remove_budget个点的索引
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(torch.logical_or(prune_mask, entropy_mask), selected_pts_mask)
            # final_prune = torch.logical_and(prune_mask,selected_pts_mask)
            self.prune_points(final_prune)
        torch.cuda.empty_cache()

    def densify_and_clone_bayesian(self, grads, budget, filter):
        # todo // Ours Bayesian-based cloning: clone low uncertainty gaussians
        # 只保留需要clone的点，其余点的梯度设为 0
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]  # 当前点云的总点数
        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device=devices.train_device())  # 初始化全 False 的掩码
        # 基于梯度采样点
        # 按梯度概率采样 budget 个点
        # torch.multinomial根据 grads的概率分布采样 budget个点（梯度越大的点越容易被选中）
        sampled_indices = torch.multinomial(grads, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True  # 标记选中的点

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self.inverse_opacity_activation(
            1 - torch.sqrt(1 - self.opacity_activation(self._opacity[selected_pts_mask])))
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_survival_logit = self._survival_logit[selected_pts_mask]
        # new_survival_logit = self.inverse_survival_prob_activation(
        #     self.survival_prob_activation(self._survival_logit[selected_pts_mask]) * 0.8)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_survival_logit)

    def densify_and_split_taming(self, grads, budget, filter, N=2):
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]

        padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
        padded_importance[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device=devices.train_device())

        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True
        # 筛选父点尺度并扩展尺度张量
        # stds是子点的尺度基准.表示每个子点在XYZ方向的标准差
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        # 初始化均值（零偏移） [N*K, 3]
        means = torch.zeros((stds.size(0), 3), device=devices.train_device())
        # 高斯采样生成偏移量  [N*K, 3]
        samples = torch.normal(mean=means, std=stds)
        # 构建父点旋转矩阵并扩展的核心步骤。它的作用是将父点的旋转（四元数）转换为旋转矩阵，
        # 并为每个父点复制N份，以便后续将子点的局部偏移量对齐到父点的旋转坐标系中
        # self._rotation：所有高斯点的旋转参数（四元数），形状为 [M, 4]（M是总点数）
        # 旋转矩阵，形状 [N*M, 3, 3]（M是选中的父点数）
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # 生成子点的所有属性（位置、尺度、旋转、颜色等）并合并到点云中
        # 计算子点的世界坐标
        # 批量矩阵乘法，将局部偏移旋转到世界坐标系，输出 [N*K, 3, 1]----->[N*K, 3]
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 计算子点的尺度 0.8 * N：缩放因子（经验值） 确保N个子点的总覆盖范围 ≈ 父点（防止过度重叠）
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        # 继承父点的其他属性
        # 旋转
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        # new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_opacity = self.inverse_opacity_activation(
            1 - torch.sqrt(1 - self.opacity_activation(self._opacity[selected_pts_mask]))).repeat(2, 1)
        new_survival_logit = self.inverse_survival_prob_activation(
            self.survival_prob_activation(self._survival_logit[selected_pts_mask]) * 0.8).repeat(2, 1)
        # new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_survival_logit)
        # 剪枝父点 prune_filter标记所有父点（selected_pts_mask）为 True，子点为 False
        # 调用 prune_points删除父点（保留子点）
        # 用更精细的子点替代原始大尺寸点，提升局部细节。
        # 分裂（Split）操作后的剪枝（Prune）步骤，
        # 目的是删除被分裂的父点，仅保留新生成的子点，从而维持点云的总数稳定并优化几何结构
        prune_filter = torch.cat(
            # 计算需剪枝的父点数：
            # selected_pts_mask.sum()得到被选中的父点数量 K
            # 生成子点占位符：
            # torch.zeros(K * N, dtype=bool)创建长度为 K*N的全 False张量，表示新子点不应被剪枝
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=devices.train_device(), dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_prune_GEO_Improved(self, scores, min_opacity, budget, opt, iteration, extent, iter_num,
                                       post_survival_prob):
        """
        改进的密度控制和剪枝方法，增强了特征感知分裂（FAS）的集成
        """
        # 计算平均梯度 即计算每个高斯点的平均屏幕空间梯度幅值（累积梯度除以更新次数）
        grad_vars = self.xyz_gradient_accum / self.denom
        # 将无效值（如除零）置为0，避免后续操作崩溃
        grad_vars[grad_vars.isnan()] = 0.0
        # 默认梯度阈值，用于决定是否增加高斯分布
        min_grad = opt.densify_grad_threshold
        # todo // Ours
        # Compute survival probability (for Bayesian framework)
        # survival_prob = self.survival_prob
        entropy = self.survival_entropy(post_survival_prob).squeeze(-1)  # 形状 [N]
        # Modify densification criteria with Bayesian uncertainty
        # Create modified gradient norm that combines gradient and uncertainty
        # 计算梯度范数 梯度范数越大，表示该点在训练中需要更多优化（如位于边缘或高变化区域）
        grad_norm = torch.linalg.norm(grad_vars, dim=-1)  # 形状 [N]
        # 联合梯度与熵的修正度量
        if iteration <= 7000:
            modified_grad = grad_norm * (1 + 2 * entropy)
        elif iteration <= 13500:
            modified_grad = grad_norm * (1 + 1.5 * entropy)
        else:
            # modified_grad = grad_norm * survival_prob.squeeze(-1)
            modified_grad = grad_norm * (1 + entropy)
        # 动态控制高斯点分裂（Splitting）和剪枝（Pruning）的核心逻辑
        # 主要功能是根据梯度幅值和当前点云状态，调整分裂预算（budget）和最小梯度阈值（min_grad）
        if iteration > 13500:
            min_grad = min_grad / 1.5  # 放宽梯度阈值

        # 梯度合格点筛选
        # torch.norm(grad_vars, dim=-1)：计算每个点的梯度L2范数（形状 [N]）
        # torch.where(condition, True, False)：生成布尔掩码，标记梯度幅值 >= min_grad的点
        # grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= min_grad, True, False)
        # grad_qualifiers = torch.where(modified_grad >= min_grad, True, False)
        # grad_qualifiers = torch.where(modified_grad >= min_grad, True, False)
        grad_qualifiers = torch.where(modified_grad >= min_grad, True, False)
        clone_qualifiers = (torch.max(self.get_scaling, dim=1).values < self.percent_dense * extent)
        split_qualifiers = (torch.max(self.get_scaling, dim=1).values >= self.percent_dense * extent)
        # 克隆与分裂条件
        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers)

        total_clones = torch.sum(all_clones).item()
        total_splits = torch.sum(all_splits).item()
        # 计算当前点数和预算的关系，确保不超过预算
        # total_sum = torch.sum(grad_qualifiers).item()  # 合格点数
        # 获取当前所有高斯点
        curr_points = len(self.get_xyz)
        # 动态分配克隆（clone）和分裂（split）操作的预算，确保在增加高斯点数量时不超过总点数限制（budget），同时合理分配克隆和分裂的配额
        budget = min(budget, total_clones + total_splits + curr_points)
        # 克隆与分裂预算分配 克隆和分裂的预算与其候选点数成正比
        clone_budget = ((budget - curr_points) * total_clones) // (total_clones + total_splits)
        # split_budget = ((budget - curr_points) * total_splits) // (total_clones + total_splits)
        split_budget = budget - curr_points - clone_budget
        self.densify_and_clone_bayesian(scores.clone(), clone_budget, all_clones)
        # 动态控制高斯点分裂（Splitting）和剪枝（Pruning）的核心逻辑
        # 主要功能是根据预算和当前点云状态，执行高斯点分裂和剪枝操作
        # 首先尝试特征感知分裂策略
        self.feature_aware_split_vector(scores.clone(), split_budget, all_splits,
                                        opt.split_distance, opt.opacity_reduction)
        # self.densify_and_split_taming(scores.clone(), split_budget, all_splits)
        # 删除透明度低于 min_opacity的点（如 opacity < 0.005），这些点对渲染贡献极小
        opacity_mask = (self.get_opacity < min_opacity).squeeze()
        # big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # survival_prob_down = (self.survival_prob < 0.1).squeeze()
        # prune_mask = torch.logical_or(opacity_mask,  survival_prob_down)
        prune_mask = opacity_mask
        # if iteration < 14500:
        #     self.prune_points(prune_mask)
        remove_budget = int(0.5 * torch.sum(prune_mask).item())
        if remove_budget > 0 and iter_num < 27:
            entropy = self.survival_entropy(self.survival_prob).squeeze(-1)
            entropy_mask = entropy <= 0.05
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            # 将重要性分数转换为剪枝概率
            # 1 / (scores + 1e-6)：分数越低 → 概率越高（更可能被剪枝）
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores)
            # 基于重要性的采样
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device=devices.train_device())
            # torch.multinomial:按概率分布（padded_importance）随机采样remove_budget个点的索引
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(torch.logical_or(prune_mask, entropy_mask), selected_pts_mask)
            # final_prune = torch.logical_and(prune_mask,selected_pts_mask)
            self.prune_points(final_prune)
        torch.cuda.empty_cache()

    def feature_aware_split_vector(self, grads, budget, filter, split_distance, opacity_reduction):
        """
        优化的特征感知分裂函数 - 只考虑每个高斯球的尺度和颜色，仅进行单向分裂
        不依赖于相邻点的位置关系，每次只朝着一个方向分裂出两个子点
        """
        # 过滤无效点
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]
        device = devices.train_device()

        # 重要性评分处理
        padded_importance = torch.zeros(n_init_points, dtype=torch.float32, device=device)
        padded_importance[:grads.shape[0]] = grads.squeeze()

        # 调整预算
        num = (padded_importance > 0).sum().item()
        if budget > num:
            budget = num

        # 采样候选点
        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device=device)
        selected_pts_mask[sampled_indices] = True

        # 获取父点参数
        parent_xyz = self.get_xyz[selected_pts_mask]
        parent_scaling = self.get_scaling[selected_pts_mask]
        parent_dc = self._features_dc[selected_pts_mask].squeeze(1)

        # 设置阈值
        geo_threshold = 2.0
        color_threshold = 0.2

        # 计算长轴信息
        max_values, max_indices = torch.max(parent_scaling, dim=1, keepdim=True)

        # 计算颜色复杂度
        color_std = torch.std(parent_dc, dim=1)

        # 计算长轴与其他轴的比值
        mask = torch.zeros_like(parent_scaling, dtype=torch.bool).scatter(1, max_indices, True)
        inv_mask = ~mask
        other_axes_sum = torch.sum(parent_scaling * inv_mask, dim=1, keepdim=True)
        other_axes_avg = other_axes_sum / 2
        axis_ratio = max_values.squeeze() / (other_axes_avg.squeeze() + 1e-10)

        # 确定分裂方向
        split_directions = max_indices.squeeze()  # 默认沿长轴

        # 颜色主导的点确定颜色主导方向
        color_dominant = (axis_ratio <= geo_threshold) & (color_std > color_threshold)
        if color_dominant.any():
            color_indices = torch.nonzero(color_dominant).squeeze(1)
            color_intensity = torch.abs(parent_dc[color_dominant])
            color_max_indices = torch.argmax(color_intensity, dim=1)
            split_directions[color_indices] = color_max_indices

        # 转换为列向量以便scatter操作
        split_directions = split_directions.unsqueeze(1)

        # 创建分裂方向掩码
        dir_mask = torch.zeros_like(parent_scaling, dtype=torch.bool).scatter(1, split_directions, True)

        # 计算偏移量
        samples = torch.zeros_like(parent_scaling)
        samples[dir_mask] = parent_scaling[dir_mask] * 3 * split_distance

        # 计算双向偏移
        x1 = torch.cat([samples, -samples], dim=0)

        # 应用旋转
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_xyz = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + parent_xyz.repeat(2, 1)

        # 生成子点参数
        # rate_w = 0.5
        # rate_h = math.sqrt(2) / 2
        rate_w = 1 - split_distance
        rate_h = math.sqrt(1 - split_distance * split_distance)

        new_scaling_val = parent_scaling.clone()
        new_scaling_val[dir_mask] *= rate_w / rate_h
        new_scaling_val = new_scaling_val * rate_h
        new_scaling = self.scaling_inverse_activation(new_scaling_val.repeat(2, 1))

        # 透明度、旋转等参数复制与调整
        new_opacity = self.inverse_opacity_activation(
            1 - torch.sqrt(1 - self.opacity_activation(self._opacity[selected_pts_mask]) * opacity_reduction)).repeat(2,
                                                                                                                      1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_survival_logit = self.inverse_survival_prob_activation(
            self.survival_prob_activation(self._survival_logit[selected_pts_mask]) * 0.8).repeat(2, 1)

        # # 添加扰动
        # perturb_scale = 0.01
        # new_xyz += torch.randn_like(new_xyz) * perturb_scale

        # 添加新点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                   new_rotation, new_survival_logit)

        # 剪枝父点
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * budget, device=device, dtype=bool)))
        self.prune_points(prune_filter)
        return selected_pts_mask

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 梯度幅值累积
        # 获取梯度：
        # viewspace_point_tensor.grad[update_filter, :2]提取被标记点（update_filter）的屏幕空间梯度（[K, 2]，K为可见点数）。
        # 梯度来源：渲染损失（如RGB L1）反向传播到屏幕坐标的梯度
        # 计算每个点梯度的L2范数（即幅值），形状从 [K, 2]→ [K, 1]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        # 更新次数统计
        # 统计每个高斯点被更新的次数（即参与训练的迭代步数）
        # 后续计算梯度均值时作为分母，避免少数高梯度值点被过度分裂。
        self.denom[update_filter] += 1

    def add_densification_stats_abs(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    """
    高斯点分裂（Splitting）的核心实现
    具体为沿高斯点的主轴（长轴）方向分裂
    目的是在预算限制内增加点云密度，提升几何细节重建
    """

    def long_axis_split(self, grads, budget, filter, split_distance, opacity_reduction):
        """
        Long axis splitting of Gaussians
        Args:
            grads: 实际是边缘感知评分-Edge Awareness Score
            budget: 计算剩余可用预算
            filter: 梯度合格点掩码，标记允许分裂的高斯点
            split_distance: 分裂时子点与父点的偏移距离系数（如 0.45）
            opacity_reduction: 子高斯的透明度设置为原始透明度的0.6
        Returns:
        """
        # 过滤无效点,将未通过筛选（filter=False）的高斯点评分置零，排除其分裂资格
        grads[~filter] = 0
        n_init_points = self.get_xyz.shape[0]
        # 为了处理可能的点数变化，创建一个与初始点数相同长度的零张量
        padded_importance = torch.zeros((n_init_points), dtype=torch.float32)

        padded_importance[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device=devices.train_device())
        # 预算调整
        num = (padded_importance > 0).sum().item()
        # 若预算budget超过有效候选点数 num，则按实际有效数分配
        if budget > num:
            budget = num
        # 采样分裂点
        # torch.multinomial:从多项分布中抽取样本。它的主要作用是根据给定的概率分布进行随机采样，返回的是采样得到的索引
        # 按评分padded_importance的概率分布，无重复采样 budget个点索引 高评分点更易被选中
        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True
        # 筛选父点尺度并扩展尺度张量
        # stds是子点的尺度基准.表示每个子点在XYZ方向的标准差
        stds = self.get_scaling[selected_pts_mask]  # 选中点的尺度 [budget,3]
        # 每个点在指定维度上的最大值（形状 [N, 1]）
        # 对应最大值的索引位置（形状 [N, 1]）
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)  # 找每点最长轴
        # 标记每个样本中最大值所在的位置
        # .scatter(1, max_indices, True)是关键的散射操作：
        # 第一个参数 1表示沿着第1维度(即列方向)进行散射
        # max_indices是一个包含每行最大值索引的张量
        # True是要填充的值
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)
        # 放大长轴尺度（*3）增强分裂后的子点偏移效果
        samples = stds * mask * 3  # 仅在长轴方向保留尺度

        # 子点位置计算
        reduction = opacity_reduction
        rate = split_distance
        x1 = samples * rate  # 长轴方向偏移量 [budget,3]
        rate_w = 1 - rate
        rate_h = math.sqrt(1 - rate * rate)
        # 计算长轴方向的双向偏移（父点两侧各一个子点）
        x1 = torch.cat([x1, -x1], dim=0)  # 正负双向偏移 [2*budget,3]
        # 通过旋转矩阵 rots将偏移量转换到世界坐标系。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        # 子点位置 = 父点位置 + 旋转后的偏移量
        new_xyz = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
        # 子点尺度 = 父点尺度在长轴方向缩小 rate_w，其他方向放大 rate_h
        # 尺度 (new_scaling)：
        # 长轴尺度按 rate_w收缩（rate_w = 1 - split_distance）。
        # 其他轴尺度按 rate_h收缩（rate_h = sqrt(1 - split_distance²)）。
        # 通过 scaling_inverse_activation映射回原始参数空间。
        new_scaling = self.scaling_inverse_activation(
            stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(2, 1) * rate_h)
        # 子点透明度 = 父点透明度 * reduction (0.6)
        new_opacity = self.inverse_opacity_activation(
            1 - torch.sqrt(1 - self.opacity_activation(self._opacity[selected_pts_mask]) * reduction)).repeat(2, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_servival_logit = self._survival_logit[selected_pts_mask].repeat(2, 1)
        # 将新点参数合并到优化器，并更新模型状态
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_servival_logit)

        # 剪枝父点 prune_filter标记所有父点（selected_pts_mask）为 True，子点为 False
        # 调用 prune_points删除父点（保留子点）
        # 用更精细的子点替代原始大尺寸点，提升局部细节。
        # 分裂（Split）操作后的剪枝（Prune）步骤，
        # 目的是删除被分裂的父点，仅保留新生成的子点，从而维持点云的总数稳定并优化几何结构
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device=devices.train_device(), dtype=bool)))
        self.prune_points(prune_filter)
        return selected_pts_mask
