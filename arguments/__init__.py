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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


"""
用于管理命令行参数的解析和分组。
它的主要功能是将类的属性自动转换为命令行参数，并提供一个方法来提取这些参数
"""


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # 球谐函数阶数，默认为3
        self.sh_degree = 3
        # 源路径
        self._source_path = ""
        # 模型路径
        self._model_path = ""
        # 图像路径
        self._images = "images"
        # 分辨率（默认-1表示未设置）
        self._resolution = -1
        # 背景是否为白色
        self._white_background = False
        # 是否为评估模式
        # self.eval = True
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    # 从输入参数中提取配置
    def extract(self, args):
        # 首先调用父类的extract方法获取参数组
        g = super().extract(args)
        # 然后将源路径转换为绝对路径
        g.source_path = os.path.abspath(g.source_path)
        return g


"""
用于管理与3D高斯泼溅（3D Gaussian Splatting）渲染管线（Pipeline）相关的参数。
这个类的作用是控制渲染过程中的一些计算方式、调试选项和图像后处理行为
"""
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        # SH(球谐函数)计算方式
        # 是否使用Python代码计算球谐函数（Spherical Harmonics, SH）。
        # 如果False，可能使用CUDA加速实现。
        self.convert_SHs_python = False# 3D协方差矩阵计算方式
        # 是否使用Python代码计算3D高斯分布的协方差矩阵（Covariance Matrix）。
        # 如果False，可能使用CUDA加速实现。
        self.compute_cov3D_python = False
        # 调试与后处理
        # 是否启用调试模式，可能会输出额外的日志或中间结果
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        """训练迭代控制"""
        # 总训练迭代次数
        self.iterations = 30_000
        # 每隔多少步执行一次高斯分布的"致密化"（densification）
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        """学习率（Learning Rate）相关"""
        # self.position_lr_init = 0.00004
        # self.position_lr_final = 0.000002
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.survival_lr_init = 0.0000016
        self.survival_lr_final = 0.0000001
        self.survival_lr_delay_mult = 0.1
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # 特征（如颜色）学习率
        self.feature_lr = 0.0025
        # 不透明度（opacity）学习率
        self.opacity_lr = 0.025
        # 高斯分布缩放（scaling）学习率
        self.scaling_lr = 0.005
        # 高斯分布旋转（rotation）学习率
        self.rotation_lr = 0.001
        """高斯分布优化控制"""
        # 控制高斯分布的稀疏/致密程度
        self.percent_dense = 0.01
        # DSSIM（结构相似性）损失的权重
        self.lambda_dssim = 0.2
        # 每隔多少步重置不透明度（opacity）
        self.opacity_reset_interval = 3000
        # 梯度阈值，用于决定是否增加高斯分布
        # # 0.0002 3dgs
        # self.densify_grad_threshold = 0.0003
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        # 优化器类型（如 "adam", "sgd"）
        self.optimizer_type = "default"
        # todo IGS新添加参数
        self.shfeature_lr = 0.005
        self.budget = 3000_000
        # split_distance: 分裂时子点与父点的偏移距离系数（如 0.45）
        self.split_distance = 0.45
        # opacity_reduction: 子高斯的透明度设置为原始透明度的0.6
        self.opacity_reduction = 1.0
        # todo // Ours bayesian
        self.use_semantic = False
        self.lambda_kl = 0.1  # KL divergence loss weight
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
