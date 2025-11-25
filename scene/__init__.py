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

import os
import random
import numpy as np
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

"""
3D高斯泼溅（3D Gaussian Splatting）场景管理的核心模块,负责加载数据、管理相机、初始化/保存高斯模型
1. 核心功能
场景初始化:从COLMAP或Blender数据加载场景（点云、相机参数）。
高斯模型管理:初始化或加载预训练的 GaussianModel，处理点云数据。
相机管理:存储训练/测试相机，支持多分辨率缩放和随机打乱。
模型保存:保存高斯模型状态（PLY文件）和曝光参数（JSON文件）
"""


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """
        :param args: 包含模型路径、数据路径等配置
        :param gaussians: 高斯模型实例。
        :param load_iteration:加载指定迭代次数的模型（-1表示最新迭代）。
        :param shuffle:是否打乱相机顺序。
        :param resolution_scales:多分辨率相机配置（如 [1.0, 0.5]）
        """
        self.model_path = args.model_path
        # 标记当前场景是否从某个训练迭代（iteration）的检查点（checkpoint）加载
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # 自动识别和加载不同格式的3D场景数据
        # 根据输入路径下的文件结构，自动判断场景数据是COLMAP格式还是Blender格式，并调用对应的加载函数
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # class SceneInfo(NamedTuple):
            #     point_cloud: BasicPointCloud
            #     train_cameras: list
            #     test_cameras: list
            #     nerf_normalization: dict
            #     ply_path: str
            #     is_nerf_synthetic: bool
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        #  将3D场景的点云数据和相机参数保存到指定目录，用于后续的训练或可视化
        #  当self.loaded_iter为 False或 None时执行（表示未加载预训练模型）
        if not self.loaded_iter:
            # 从 scene_info.ply_path复制到目标路径 self.model_path/input.ply。
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            # 保存相机参数：将所有相机（训练集 + 测试集）转换为JSON格式，保存到self.model_path/cameras.json。
            json_cams = []
            # camlist包含所有相机（先测试集，后训练集）
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)  # 添加测试相机
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)  # 添加训练相机
            for id, cam in enumerate(camlist):
                # 将单个CameraInfo对象转为字典
                json_cams.append(camera_to_JSON(id, cam))  # 假设 camera_to_JSON 返回字典
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                # 将相机参数列表（json_cams）以 JSON 格式保存到文件中
                # json.dump：将 Python 对象序列化为JSON格式并写入文件
                json.dump(json_cams, file)
        #  对训练集和测试集的相机进行随机打乱（Shuffle），以增强数据多样性和训练鲁棒性
        if shuffle:
            # 在多分辨率训练中，不同分辨率的数据流需保持相同的随机顺序
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            # self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("baseline extent: ", scene_info.nerf_normalization["radius"])
        # sfm = np.asarray(scene_info.point_cloud.points)
        # my_extent = own_distance(sfm, center=-scene_info.nerf_normalization["translate"])  # 4.1
        # print("proposed extent: ", my_extent)

        # self.cameras_extent = np.max([my_extent, scene_info.nerf_normalization["radius"]])
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 根据是否加载预训练模型，选择初始化高斯点云的方式
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        # 根据是否加载预训练模型，选择初始化高斯点云的方式
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            # scene_info.point_cloud：COLMAP生成的稀疏点云（含坐标、颜色）
            # self.cameras_extent：将点云坐标归一化到 [-1, 1]范围内，提升训练稳定性
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
# Corrected Scene-Extent
# Average distance between all SFM-input-points (points) and the average camera position (center).

def own_distance(points, center=[]):
    if len(center) != 3:
        print("incorrect center")
        center = np.mean(points, axis=0, keepdims=True)
    dist = np.linalg.norm(points - center, axis=1, keepdims=True)
    # max_distance = np.max(dist) * 1.1
    # variance = np.var(dist)
    # 平均距离计算 对所有点的距离取均值，反映点云围绕中心点的平均分布半径
    mean_dist = np.mean(dist)
    return mean_dist