#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo 数据集 DA3 推理脚本

支持批量推理 Waymo 多帧多相机数据，输出绝对深度和3D点云
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# 导入 DA3（从已安装的包中导入）
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.export.glb import export_to_glb
from parse_waymo_calibration import parse_waymo_calibration


# ========== 配置参数 ==========
# 数据路径（云端路径，需要修改）
DATA_ROOT = "/home/zyc/street_crafter/data/waymo/049"
IMAGES_DIR = Path(DATA_ROOT) / "images"

# 模型配置
MODEL_PATH = "/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"

# 推理配置
CAMERA_IDS = [0, 1, 2, 3, 4]  # Waymo 5个相机
FRAME_RANGE = (0, 10)  # 推理的帧范围 (start, end)，None表示所有帧
UNDISTORT_IMAGES = True  # 是否去畸变

# 输出配置
OUTPUT_DIR = Path("output/waymo_inference")
EXPORT_GLB = True  # 是否导出3D点云
GLB_CONF_THRESH_PERCENTILE = 5.0  # 置信度百分位阈值（越低保留越多点）
GLB_NUM_MAX_POINTS = 5_000_000  # 最大点云数量

# ========== 工具函数 ==========

def get_waymo_image_path(images_dir, frame_id, camera_id):
    """
    获取 Waymo 图像路径

    Args:
        images_dir: 图像目录
        frame_id: 帧ID
        camera_id: 相机ID

    Returns:
        Path: 图像文件路径
    """
    # Waymo 格式: XXXXXX_Y.png
    filename = f"{frame_id:06d}_{camera_id}.png"
    return images_dir / filename


def get_all_frames(images_dir, camera_ids):
    """
    获取所有可用的帧ID

    Args:
        images_dir: 图像目录
        camera_ids: 相机ID列表

    Returns:
        List[int]: 排序后的帧ID列表
    """
    # 从第一个相机获取所有帧
    first_cam_id = camera_ids[0]
    pattern = f"*_{first_cam_id}.png"

    frame_ids = []
    for img_path in sorted(images_dir.glob(pattern)):
        # 从文件名提取帧号
        frame_str = img_path.stem.split('_')[0]
        frame_id = int(frame_str)
        frame_ids.append(frame_id)

    return sorted(frame_ids)


def visualize_waymo_frame(prediction, camera_names, frame_id, save_dir):
    """
    可视化单帧的多相机深度图

    Args:
        prediction: DA3 prediction 对象（包含 processed_images 和 depth）
        camera_names: List of camera names
        frame_id: 帧ID
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 使用 DA3 处理后的图像（与深度图尺寸匹配）
    processed_images = prediction.processed_images  # (N, H, W, 3) uint8
    depth_maps = prediction.depth  # (N, H, W)

    for i, (img, depth, cam_name) in enumerate(zip(processed_images, depth_maps, camera_names)):
        # 深度彩色可视化
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_INFERNO
        )

        # 转换为RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        # 拼接原图和深度图（现在尺寸匹配了）
        vis = np.hstack([img, depth_colored])

        # 保存
        output_file = save_dir / f"frame_{frame_id:06d}_{cam_name}.jpg"
        Image.fromarray(vis).save(output_file, quality=90)

    print(f"  - 可视化已保存到: {save_dir}")


# ========== 主推理流程 ==========

def main():
    print("=" * 70)
    print("Waymo DA3 推理")
    print("=" * 70)
    print(f"数据目录: {DATA_ROOT}")
    print(f"图像目录: {IMAGES_DIR}")
    print(f"相机数量: {len(CAMERA_IDS)}")
    print(f"去畸变: {UNDISTORT_IMAGES}")

    # 1. 解析标定
    print("\n" + "-" * 70)
    print("步骤 1/5: 解析相机标定")
    print("-" * 70)

    intrinsics, extrinsics, distortions, camera_names = parse_waymo_calibration(
        DATA_ROOT, CAMERA_IDS
    )

    # 2. 获取所有帧
    print("\n" + "-" * 70)
    print("步骤 2/5: 扫描图像")
    print("-" * 70)

    all_frames = get_all_frames(IMAGES_DIR, CAMERA_IDS)
    print(f"找到 {len(all_frames)} 帧")

    # 确定推理范围
    if FRAME_RANGE is not None:
        start_idx, end_idx = FRAME_RANGE
        frames_to_process = all_frames[start_idx:end_idx]
        print(f"推理范围: 帧 {start_idx} - {end_idx-1} (共 {len(frames_to_process)} 帧)")
    else:
        frames_to_process = all_frames
        print(f"推理所有 {len(frames_to_process)} 帧")

    # 3. 加载模型
    print("\n" + "-" * 70)
    print("步骤 3/5: 加载 DA3 模型")
    print("-" * 70)

    model = DepthAnything3.from_pretrained(MODEL_PATH)
    print(f"模型已加载: {MODEL_PATH}")

    # 4. 批量推理
    print("\n" + "-" * 70)
    print("步骤 4/5: 批量推理")
    print("-" * 70)

    for frame_idx, frame_id in enumerate(frames_to_process):
        print(f"\n{'='*70}")
        print(f"处理帧 {frame_idx+1}/{len(frames_to_process)}: Frame ID {frame_id}")
        print(f"{'='*70}")

        # 加载多相机图像
        images = []
        images_for_inference = []

        for cam_id in CAMERA_IDS:
            img_path = get_waymo_image_path(IMAGES_DIR, frame_id, cam_id)

            if not img_path.exists():
                print(f"警告: 图像不存在，跳过: {img_path}")
                continue

            # 读取图像
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            images.append(img_np)

            # 去畸变
            if UNDISTORT_IMAGES:
                K = intrinsics[cam_id]
                D = distortions[cam_id]
                undist_img = cv2.undistort(img_np, K, D)
                images_for_inference.append(undist_img)
            else:
                images_for_inference.append(img_np)

        if len(images) != len(CAMERA_IDS):
            print(f"警告: 帧 {frame_id} 相机图像不完整，跳过")
            continue

        print(f"  - 加载了 {len(images)} 张图像")

        # 推理
        print("  - 开始推理...")
        try:
            prediction = model.inference(
                image=images_for_inference,
                intrinsics=intrinsics,
                extrinsics=extrinsics
            )

            print(f"  - 推理完成")
            print(f"    - 深度图: {prediction.depth.shape}")
            print(f"    - 置信度: {prediction.conf.shape}")

        except Exception as e:
            print(f"错误: 推理失败")
            import traceback
            traceback.print_exc()
            continue

        # 保存深度图
        frame_output_dir = OUTPUT_DIR / f"frame_{frame_id:06d}"
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 NPZ
        depth_file = frame_output_dir / "depths.npz"
        np.savez_compressed(
            depth_file,
            depth=prediction.depth,
            conf=prediction.conf,
            camera_names=camera_names,
            frame_id=frame_id
        )
        print(f"  - 深度已保存: {depth_file}")

        # 可视化
        visualize_waymo_frame(
            prediction=prediction,
            camera_names=camera_names,
            frame_id=frame_id,
            save_dir=frame_output_dir / "visualizations"
        )

        # 导出点云
        if EXPORT_GLB:
            print("  - 导出3D点云...")
            try:
                glb_path = export_to_glb(
                    prediction=prediction,
                    export_dir=str(frame_output_dir / "glb"),
                    conf_thresh_percentile=GLB_CONF_THRESH_PERCENTILE,
                    num_max_points=GLB_NUM_MAX_POINTS,
                    show_cameras=True,
                    camera_size=0.05
                )
                print(f"  - 点云已保存: {glb_path}")
            except Exception as e:
                print(f"  - 点云导出失败: {e}")
                import traceback
                traceback.print_exc()

        # 深度统计
        print("\n  深度统计:")
        for i, (depth, cam_name) in enumerate(zip(prediction.depth, camera_names)):
            valid_depth = depth[np.isfinite(depth) & (depth > 0)]
            if len(valid_depth) > 0:
                print(f"    {cam_name}: "
                      f"min={valid_depth.min():.2f}m, "
                      f"max={valid_depth.max():.2f}m, "
                      f"mean={valid_depth.mean():.2f}m, "
                      f"median={np.median(valid_depth):.2f}m")

    # 5. 完成
    print("\n" + "=" * 70)
    print("步骤 5/5: 推理完成")
    print("=" * 70)
    print(f"处理了 {len(frames_to_process)} 帧")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n输出文件：")
    print("  - output/waymo_inference/frame_XXXXXX/depths.npz (深度图)")
    print("  - output/waymo_inference/frame_XXXXXX/visualizations/ (可视化)")
    print("  - output/waymo_inference/frame_XXXXXX/glb/scene.glb (3D点云)")


if __name__ == "__main__":
    main()
