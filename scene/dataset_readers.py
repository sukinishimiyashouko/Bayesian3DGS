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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    # def get_center_and_diag(cam_centers):
    #     cam_centers = np.hstack(cam_centers)
    #     avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    #     center = avg_cam_center
    #     dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    #     diagonal = np.max(dist)
    #     return center.flatten(), diagonal
    def get_center_and_diag(cam_centers):
        # 水平堆叠所有相机中心坐标（np.hstack）
        cam_centers = np.hstack(cam_centers)  # 形状从 [N,3,1] 转为 [3,N]
        mean_cam_center = np.mean(cam_centers, axis=1, keepdims=True)  # 计算几何中心 [3,1]
        center = mean_cam_center
        # 计算每个相机中心到场景几何中心的欧氏距离，并保持输出的维度结构
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        # 计算距离的中位数和标准差(可选)
        median_dist = np.median(dist)
        std_dist = np.std(dist)
        # 使用中位数 + 标准差乘子作为最终范围(可选)
        extent = median_dist + 2 * std_dist
        diagonal = np.max(dist)  # 最大距离（即包围球半径）
        return center.flatten(), diagonal, extent

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal, extent = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # translate：将场景中心移动到原点所需的平移向量。
    # radius：场景的包围球半径（稍大于最大相机距离）
    return {"translate": translate, "radius": radius, "extent": extent}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


"""
从COLMAP格式的3D重建场景中读取关键数据（包括相机参数、点云、深度信息等），并将其封装为统一的 SceneInfo对象
输出：
SceneInfo对象，包含：1.点云数据（PLY格式）2.训练集和测试集的相机参数 3.NeRF所需的场景归一化参数 4.其他元数据（如深度信息、文件路径等）
"""


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        # 尝试读取二进制文件（COLMAP默认格式）
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        # BaseImage = collections.namedtuple(
        #     "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        # Camera = collections.namedtuple(
        #     "Camera", ["id", "model", "width", "height", "params"])
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    # 读取并整理COLMAP格式的相机信息，包括相机参数、图像路径、深度图路径等，并按图像名称排序
    # 确定图像目录
    reading_dir = "images" if images == None else images
    """
        参数名-类型-作用
        cam_extrinsics--dict--相机外参（位姿），通常来自 images.bin或 images.txt。
        cam_intrinsics--dict--相机内参（焦距、畸变等），来自 cameras.bin或 cameras.txt。
        images_folder--str--图像文件的实际路径（如 path/images/）。
        """
    # cam_infos_unsorted是一个未排序的相机信息列表
    """
    class CameraInfo(NamedTuple):
         uid: int
         R: np.array
         T: np.array
         FovY: np.array
         FovX: np.array
         depth_params: dict
         image_path: str
         image_name: str
         depth_path: str
         width: int
         height: int
         is_test: bool
    """
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    # 根据相机对应的图像名称（image_name）字母顺序排序
    # 避免修改原始列表并指定排序依据为 CameraInfo对象的image_name字段
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    # 划分训练/测试集
    # 评估模式（eval=True）时划分测试集。
    # 360度场景按固定间隔（llffhold=8）采样测试相机。
    # 其他场景从 test.txt读取测试相机列表
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # 计算NeRF（或3D高斯泼溅）训练所需的场景归一化参数，确保模型在标准化的坐标空间中训练
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)  # 尝试读取二进制文件
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)  # 失败则读取文本文件
        storePly(ply_path, xyz, rgb)  # 保存为PLY格式
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # class SceneInfo(NamedTuple):
    #     point_cloud: BasicPointCloud
    #     train_cameras: list
    #     test_cameras: list
    #     nerf_normalization: dict
    #     ply_path: str
    #     is_nerf_synthetic: bool
    # 用于封装3D场景的所有关键信息，包括点云、相机参数、归一化参数等
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
