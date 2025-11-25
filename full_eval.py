import os
from argparse import ArgumentParser
import time

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
# 跳过模型训练阶段。
parser.add_argument("--skip_training", action="store_true")
# 跳过渲染阶段，仅执行训练或指标计算
parser.add_argument("--skip_rendering", action="store_true")
# 跳过图像质量指标（如SSIM、PSNR、LPIPS）的计算
parser.add_argument("--skip_metrics", action="store_true")
# 指定输出结果的根目录路径（默认是 ./eval）
parser.add_argument("--output_path", default="/HYYJS-SSD-1/wq/Bayesian3DGS/eval_3")

paramList = {
    "bicycle": 3000_000,
    "flowers": 1500_000,
    "garden": 3000_000,
    "stump": 3000_000,
    "treehill": 1500_000,

    "bonsai": 1000_000,
    "counter": 1000_000,
    "kitchen": 1000_000,
    "room": 1000_000,

    "drjohnson": 1500_000,
    "playroom": 1000_000,

    "train": 1000_000,
    "truck": 1500_000
}


# CUDA_VISIBLE_DEVICES=0 python full_eval.py
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", type=str, default="../data_main/data")
    parser.add_argument("--tanksandtemples", "-tat", type=str, default="../data_main/data")
    parser.add_argument("--deepblending", "-db", type=str, default="../data_main/data")
    args = parser.parse_args()
if not args.skip_training:
    # --quiet : 启用静默模式，即减少或禁止程序运行时的非必要输出（如日志、进度条、调试信息等），仅保留关键结果或错误信息
    # common_args = " --quiet --eval --test_iterations -1 "
    common_args = " --eval "
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=1 python train.py -s " + source + " -i images -m " + args.output_path + "/" + scene + common_args+"--budget "+str(paramList[scene]))
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=1 python train.py -s " + source + " -i images -m " + args.output_path + "/" + scene + common_args+"--budget "+str(paramList[scene]))
    m360_timing = (time.time() - start_time) / 60.0

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=1 python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args+"--budget "+str(paramList[scene]))
    tandt_timing = (time.time() - start_time) / 60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=1 python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args+"--budget "+str(paramList[scene]))
    db_timing = (time.time() - start_time) / 60.0

with open(os.path.join(args.output_path, "timing.txt"), 'w') as file:
    file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        # os.system(
        #     "python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system(
            "python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
