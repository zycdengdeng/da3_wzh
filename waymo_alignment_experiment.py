#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo 多视角深度对齐实验脚本

测试不同的深度对齐策略来优化多视角点云的衔接质量
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import export_to_glb
from parse_waymo_calibration import parse_waymo_calibration


# ========== 配置参数 ==========
DATA_ROOT = "/home/zyc/street_crafter/data/waymo/049"
IMAGES_DIR = Path(DATA_ROOT) / "images"
MODEL_PATH = "/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"

CAMERA_IDS = [0, 1, 2, 3, 4]
TEST_FRAME = 0  # 测试单帧
UNDISTORT_IMAGES = True

# 对齐策略实验
EXPERIMENTS = [
    {
        "name": "default_global_align",
        "align_to_input_ext_scale": True,  # 默认：使用全局scale对齐
        "description": "默认策略：全局尺度对齐"
    },
    {
        "name": "no_scale_align",
        "align_to_input_ext_scale": False,  # 不强制对齐到输入scale
        "description": "不强制对齐：使用模型预测的相对位姿"
    },
]


def get_waymo_image_path(images_dir, frame_id, camera_id):
    filename = f"{frame_id:06d}_{camera_id}.png"
    return images_dir / filename


def load_frame_images(images_dir, frame_id, camera_ids, intrinsics, distortions, undistort=True):
    """加载并预处理单帧的所有相机图像"""
    images = []

    for cam_id in camera_ids:
        img_path = get_waymo_image_path(images_dir, frame_id, cam_id)
        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if undistort:
            K = intrinsics[cam_id]
            D = distortions[cam_id]
            img_np = cv2.undistort(img_np, K, D)

        images.append(img_np)

    return images


def run_experiment(experiment, model, images, intrinsics, extrinsics, output_dir):
    """运行单个对齐策略实验"""
    exp_name = experiment["name"]
    align_flag = experiment["align_to_input_ext_scale"]

    print(f"\n{'='*70}")
    print(f"实验: {exp_name}")
    print(f"描述: {experiment['description']}")
    print(f"align_to_input_ext_scale: {align_flag}")
    print(f"{'='*70}")

    # 推理
    prediction = model.inference(
        image=images,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        align_to_input_ext_scale=align_flag  # 关键参数
    )

    # 输出统计
    print(f"\n深度统计 ({exp_name}):")
    for i, depth in enumerate(prediction.depth):
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(valid_depth) > 0:
            print(f"  相机 {i}: "
                  f"min={valid_depth.min():.2f}m, "
                  f"max={valid_depth.max():.2f}m, "
                  f"mean={valid_depth.mean():.2f}m, "
                  f"std={valid_depth.std():.2f}m")

    # 导出点云
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    glb_path = export_to_glb(
        prediction=prediction,
        export_dir=str(exp_output_dir),
        conf_thresh_percentile=5.0,
        num_max_points=5_000_000,
        show_cameras=True,
        camera_size=0.05
    )

    print(f"✓ 点云已保存: {glb_path}")

    # 保存深度图
    np.savez_compressed(
        exp_output_dir / "depths.npz",
        depth=prediction.depth,
        conf=prediction.conf,
        extrinsics=prediction.extrinsics,
        intrinsics=prediction.intrinsics
    )

    return prediction


def analyze_overlap_consistency(prediction, camera_pairs=[(0, 1), (1, 2), (2, 0)]):
    """
    分析重叠区域的深度一致性（简化版）

    理论上应该通过投影找到重叠像素，这里简化为统计分析
    """
    print(f"\n{'='*70}")
    print("重叠区域一致性分析（统计估计）")
    print(f"{'='*70}")

    depths = prediction.depth

    for cam_a, cam_b in camera_pairs:
        depth_a = depths[cam_a]
        depth_b = depths[cam_b]

        # 简化：比较深度范围重叠
        valid_a = depth_a[np.isfinite(depth_a) & (depth_a > 0)]
        valid_b = depth_b[np.isfinite(depth_b) & (depth_b > 0)]

        if len(valid_a) > 0 and len(valid_b) > 0:
            mean_diff = abs(valid_a.mean() - valid_b.mean())
            std_diff = abs(valid_a.std() - valid_b.std())

            print(f"\n相机对 {cam_a}-{cam_b}:")
            print(f"  平均深度差异: {mean_diff:.3f}m")
            print(f"  标准差差异: {std_diff:.3f}m")


def main():
    print("="*70)
    print("Waymo 多视角深度对齐实验")
    print("="*70)

    # 1. 解析标定
    intrinsics, extrinsics, distortions, camera_names = parse_waymo_calibration(
        DATA_ROOT, CAMERA_IDS
    )

    # 2. 加载图像
    print(f"\n加载帧 {TEST_FRAME} 的图像...")
    images = load_frame_images(
        IMAGES_DIR, TEST_FRAME, CAMERA_IDS,
        intrinsics, distortions, UNDISTORT_IMAGES
    )
    print(f"✓ 加载了 {len(images)} 张图像")

    # 3. 加载模型
    print("\n加载 DA3 模型...")
    model = DepthAnything3.from_pretrained(MODEL_PATH)
    print("✓ 模型已加载")

    # 4. 运行实验
    OUTPUT_DIR = Path("output/waymo_alignment_experiments")
    results = {}

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment, model, images,
            intrinsics, extrinsics, OUTPUT_DIR
        )
        results[experiment["name"]] = result

        # 分析一致性
        analyze_overlap_consistency(result)

    # 5. 总结
    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n各实验结果:")
    for exp in EXPERIMENTS:
        exp_dir = OUTPUT_DIR / exp["name"]
        print(f"\n{exp['name']}:")
        print(f"  描述: {exp['description']}")
        print(f"  点云: {exp_dir / 'scene.glb'}")
        print(f"  深度: {exp_dir / 'depths.npz'}")

    print("\n建议:")
    print("1. 在Blender/MeshLab中打开两个GLB文件对比")
    print("2. 观察点云的连续性和重叠区域的衔接")
    print("3. 如果 no_scale_align 效果更好，可在主脚本中使用该参数")


if __name__ == "__main__":
    main()
