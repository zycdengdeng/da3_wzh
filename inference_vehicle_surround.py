#!/usr/bin/env python3
"""
车端周视相机 DA3 推理
针对车端环视场景的深度估计和3D重建
"""

import torch
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_vehicle_camera_params(npz_file):
    """
    加载车端相机参数文件

    Args:
        npz_file: 相机参数文件路径

    Returns:
        extrinsics, intrinsics, distortions, camera_names, image_sizes
    """
    data = np.load(npz_file, allow_pickle=True)

    extrinsics = data['extrinsics']    # (N, 4, 4)
    intrinsics = data['intrinsics']    # (N, 3, 3)
    distortions = data['distortions']  # (N, 5)
    camera_names = data['camera_names']  # (N,)
    image_sizes = data['image_sizes']  # (N, 2)

    return extrinsics, intrinsics, distortions, camera_names, image_sizes


def undistort_images(images, intrinsics, distortions, camera_names):
    """
    对图像进行去畸变处理

    Args:
        images: 图像列表
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变系数
        camera_names: 相机名称列表

    Returns:
        undistorted_images: 去畸变后的图像列表
    """
    undistorted_images = []

    for i, img in enumerate(images):
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img

        K = intrinsics[i]
        dist = distortions[i]

        # 检查是否为 camera_01 (畸变参数异常)
        if 'camera_01' in camera_names[i]:
            print(f"  相机 {i+1} ({camera_names[i]}): 畸变参数异常，跳过去畸变")
            undist_img = img_np.copy()
        else:
            # 直接使用 cv2.undistort，不使用 getOptimalNewCameraMatrix
            undist_img = cv2.undistort(img_np, K, dist)
            print(f"  相机 {i+1} ({camera_names[i]}): 去畸变完成")

        # 转换回PIL
        undistorted_images.append(Image.fromarray(undist_img))

    return undistorted_images


def visualize_surround_view(images, depth_maps, camera_names, save_dir):
    """
    可视化环视相机的深度图

    Args:
        images: 原始图像列表
        depth_maps: (N, H, W) 深度图
        camera_names: 相机名称列表
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 单独保存每个相机的结果
    for i in range(len(images)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 原始图像
        if isinstance(images[i], Image.Image):
            img = np.array(images[i])
        else:
            img = images[i]
        ax1.imshow(img)
        ax1.set_title(f'Camera {i+1}: {camera_names[i]}')
        ax1.axis('off')

        # 深度图
        depth = depth_maps[i]
        im_depth = ax2.imshow(depth, cmap='turbo', vmin=0, vmax=50)  # 车端场景0-50米
        ax2.set_title(f'Depth (Range: {depth.min():.2f}-{depth.max():.2f}m)')
        ax2.axis('off')
        plt.colorbar(im_depth, ax=ax2, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 保存
        save_path = save_dir / f'camera_{i+1:02d}_{camera_names[i]}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 相机 {i+1} 可视化已保存: {save_path}")


def main():
    """主函数"""

    # ========== 配置 ==========
    print("=" * 70)
    print("车端周视相机 DA3 深度估计")
    print("=" * 70)

    # 数据路径（请修改为您的实际路径）
    DATA_DIR = Path("/home/zyc/车端周视图")

    # 如果远程路径不存在，尝试相对路径
    if not DATA_DIR.exists():
        DATA_DIR = Path("./车端周视图")

    if not DATA_DIR.exists():
        print(f"错误: 找不到数据目录 {DATA_DIR}")
        print("请修改脚本中的 DATA_DIR 路径")
        return

    # 相机参数文件（由 parse_vehicle_calibration.py 生成）
    CAMERA_PARAMS_FILE = "./output/vehicle_surround_calibration/vehicle_camera_params.npz"

    if not Path(CAMERA_PARAMS_FILE).exists():
        print(f"错误: 找不到相机参数文件 {CAMERA_PARAMS_FILE}")
        print("请先运行: python parse_vehicle_calibration.py")
        return

    # 图像文件映射（相机ID -> 文件名）
    IMAGE_MAPPING = {
        1: 'FN.jpg',  # 前窄角
        2: 'FW.jpg',  # 前广角
        3: 'FL.jpg',  # 前左
        4: 'FR.jpg',  # 前右
        5: 'RL.jpg',  # 左后
        6: 'RR.jpg',  # 右后
        7: 'RN.jpg',  # 后视
    }

    # 模型路径
    MODEL_PATH = "/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"

    # 检查模型路径
    if not Path(MODEL_PATH).exists():
        print(f"警告: 模型路径不存在: {MODEL_PATH}")
        print(f"请检查模型文件位置")
        return

    # 输出目录
    OUTPUT_DIR = Path("./output/vehicle_surround_inference")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 推理配置
    UNDISTORT_IMAGES = True  # 车端图像需要去畸变
    PROCESS_RES = 504        # 处理分辨率

    # ========== 加载相机参数 ==========
    print("\n" + "-" * 70)
    print("加载相机参数")
    print("-" * 70)

    extrinsics, intrinsics, distortions, camera_names, image_sizes = load_vehicle_camera_params(
        CAMERA_PARAMS_FILE
    )

    print(f"✓ 已加载 {len(extrinsics)} 个相机的参数")
    for i, name in enumerate(camera_names):
        print(f"  [{i+1}] {name} - {image_sizes[i][0]}x{image_sizes[i][1]}")

    # ========== 加载图像 ==========
    print("\n" + "-" * 70)
    print("加载图像")
    print("-" * 70)

    image_paths = []
    for cam_id in IMAGE_MAPPING.keys():
        img_name = IMAGE_MAPPING[cam_id]
        img_path = DATA_DIR / img_name

        if not img_path.exists():
            print(f"警告: 找不到图像 {img_path}")
            continue

        image_paths.append(img_path)
        print(f"✓ 找到图像: {img_name}")

    if len(image_paths) != len(camera_names):
        print(f"警告: 图像数量({len(image_paths)})与相机数量({len(camera_names)})不匹配")
        return

    # 加载图像
    images = [Image.open(img_path) for img_path in image_paths]
    print(f"\n✓ 已加载 {len(images)} 张图像")

    # ========== 图像去畸变 ==========
    if UNDISTORT_IMAGES:
        print("\n" + "-" * 70)
        print("图像去畸变处理")
        print("-" * 70)

        images_undistorted = undistort_images(images, intrinsics, distortions, camera_names)

        # 保存去畸变后的图像
        undist_dir = OUTPUT_DIR / "undistorted_images"
        undist_dir.mkdir(exist_ok=True)

        for i, img in enumerate(images_undistorted):
            save_path = undist_dir / f"undistorted_{camera_names[i]}.jpg"
            img.save(save_path, quality=95)
            print(f"  保存: {save_path}")

        # 使用去畸变后的图像
        images_for_inference = images_undistorted
    else:
        images_for_inference = images

    # ========== 加载 DA3 模型 ==========
    print("\n" + "-" * 70)
    print("加载 DA3 模型")
    print("-" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        model = DepthAnything3.from_pretrained(MODEL_PATH)
        model = model.to(device=device)
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # ========== 执行推理 ==========
    print("\n" + "-" * 70)
    print("执行深度估计推理")
    print("-" * 70)

    print(f"推理配置:")
    print(f"  - 处理分辨率: {PROCESS_RES}")
    print(f"  - 使用相机参数: 是")
    print(f"  - 图像去畸变: {'是' if UNDISTORT_IMAGES else '否'}")
    print(f"  - 输出格式: NPZ (深度数据) + GLB (3D点云)")

    try:
        # 生成深度数据
        prediction = model.inference(
            image=images_for_inference,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=PROCESS_RES,
            align_to_input_ext_scale=True,
            export_dir=str(OUTPUT_DIR / "da3_output"),
            export_format="mini_npz",
        )

        print(f"\n✓ 深度推理完成!")
        print(f"  - 深度图 shape: {prediction.depth.shape}")
        print(f"  - 是否度量深度: {bool(prediction.is_metric)}")

        # 生成 3D 点云
        print(f"\n正在生成车端环视 3D 点云...")
        prediction_pointcloud = model.inference(
            image=images_for_inference,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=PROCESS_RES,
            align_to_input_ext_scale=True,
            export_dir=str(OUTPUT_DIR / "pointcloud"),
            export_format="glb",
            conf_thresh_percentile=10.0,   # 车端场景：保留90%的点
            num_max_points=3_000_000,      # 环视7个相机：300万点
            show_cameras=True,
        )

        print(f"  置信度阈值: 10% (保留90%的点)")
        print(f"  最大点数: 300万")
        print(f"\n✓ 3D 点云生成完成!")
        print(f"  - 点云文件: {OUTPUT_DIR / 'pointcloud' / 'scene.glb'}")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 结果可视化 ==========
    print("\n" + "-" * 70)
    print("生成结果可视化")
    print("-" * 70)

    visualize_surround_view(
        images=images_for_inference,  # 使用去畸变后的图像
        depth_maps=prediction.depth,
        camera_names=camera_names,
        save_dir=OUTPUT_DIR / "visualizations"
    )

    # ========== 深度统计 ==========
    print("\n" + "-" * 70)
    print("深度统计分析")
    print("-" * 70)

    for i in range(len(prediction.depth)):
        depth = prediction.depth[i]
        print(f"\n相机 {i+1} ({camera_names[i]}):")
        print(f"  深度范围: {depth.min():.2f} - {depth.max():.2f} 米")
        print(f"  平均深度: {depth.mean():.2f} 米")
        print(f"  中位数深度: {np.median(depth):.2f} 米")

    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)

    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print(f"  - 深度数据: {OUTPUT_DIR / 'da3_output' / 'prediction.npz'}")
    print(f"  - 3D 点云: {OUTPUT_DIR / 'pointcloud' / 'scene.glb'}")
    print(f"  - 环视可视化: {OUTPUT_DIR / 'visualizations' / 'camera_*.png'}")
    print(f"  - 去畸变图像: {OUTPUT_DIR / 'undistorted_images' / '*.jpg'}")

    print("\n提示:")
    print("  1. 使用 Blender/MeshLab 打开 scene.glb 查看车端环视 3D 点云")
    print("  2. 查看 visualizations/ 目录查看各相机的深度图")
    print("  3. 车端场景深度范围通常在 0-50 米")
    print("\n在线查看器: https://gltf-viewer.donmccurdy.com/")


if __name__ == "__main__":
    main()
