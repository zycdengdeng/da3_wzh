#!/usr/bin/env python3
"""
使用真实标定文件进行 DA3 推理
针对路侧四相机系统的深度估计
"""

import torch
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_camera_params(npz_file):
    """
    加载相机参数文件

    Args:
        npz_file: 相机参数文件路径

    Returns:
        extrinsics, intrinsics, distortions, camera_names
    """
    data = np.load(npz_file, allow_pickle=True)

    extrinsics = data['extrinsics']    # (N, 4, 4)
    intrinsics = data['intrinsics']    # (N, 3, 3)
    distortions = data['distortions']  # (N, 5)
    camera_names = data['camera_names']  # (N,)

    return extrinsics, intrinsics, distortions, camera_names


def undistort_images(images, intrinsics, distortions):
    """
    对图像进行去畸变处理（如果启用）

    Args:
        images: 图像列表
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变系数

    Returns:
        undistorted_images: 去畸变后的图像列表

    Note:
        直接使用 cv2.undistort(img, K, D) 而不是 getOptimalNewCameraMatrix
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

        # 直接使用 cv2.undistort，不使用 getOptimalNewCameraMatrix
        undist_img = cv2.undistort(img_np, K, dist)

        # 转换回PIL
        undistorted_images.append(Image.fromarray(undist_img))

        print(f"  图像 {i} 去畸变完成")

    return undistorted_images


def visualize_results(
    images,
    depth_maps,
    sky_masks,
    conf_maps,
    camera_names,
    save_dir
):
    """
    可视化推理结果

    Args:
        images: 原始图像列表
        depth_maps: (N, H, W) 深度图
        sky_masks: (N, H, W) 天空分割
        conf_maps: (N, H, W) 置信度图
        camera_names: 相机名称列表
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 原始图像
        if isinstance(images[i], Image.Image):
            img = np.array(images[i])
        else:
            img = images[i]
        axes[0, 0].imshow(img)
        axes[0, 0].set_title(f'Original Image - {camera_names[i]}')
        axes[0, 0].axis('off')

        # 深度图
        depth = depth_maps[i]
        im_depth = axes[0, 1].imshow(depth, cmap='turbo')
        axes[0, 1].set_title(f'Depth Map (Min: {depth.min():.2f}m, Max: {depth.max():.2f}m)')
        axes[0, 1].axis('off')
        plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 天空分割
        if sky_masks is not None:
            sky = sky_masks[i]
            axes[1, 0].imshow(sky, cmap='gray')
            axes[1, 0].set_title('Sky Segmentation')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Sky Mask', ha='center', va='center')
            axes[1, 0].axis('off')

        # 置信度图
        if conf_maps is not None:
            conf = conf_maps[i]
            im_conf = axes[1, 1].imshow(conf, cmap='viridis')
            axes[1, 1].set_title('Confidence Map')
            axes[1, 1].axis('off')
            plt.colorbar(im_conf, ax=axes[1, 1], fraction=0.046, pad=0.04)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Confidence Map', ha='center', va='center')
            axes[1, 1].axis('off')

        plt.tight_layout()

        # 保存
        save_path = save_dir / f'result_{i:02d}_{camera_names[i]}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 结果可视化已保存: {save_path}")


def create_depth_statistics_report(depth_maps, sky_masks, camera_names, save_path):
    """
    创建深度统计报告

    Args:
        depth_maps: (N, H, W) 深度图
        sky_masks: (N, H, W) 天空分割
        camera_names: 相机名称列表
        save_path: 报告保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("深度估计统计报告\n")
        f.write("=" * 70 + "\n\n")

        for i in range(len(depth_maps)):
            depth = depth_maps[i]

            f.write(f"相机 {i} ({camera_names[i]}):\n")
            f.write("-" * 70 + "\n")

            # 整体统计
            f.write(f"整体深度统计:\n")
            f.write(f"  最小值: {depth.min():.2f} 米\n")
            f.write(f"  最大值: {depth.max():.2f} 米\n")
            f.write(f"  平均值: {depth.mean():.2f} 米\n")
            f.write(f"  中位数: {np.median(depth):.2f} 米\n")
            f.write(f"  标准差: {depth.std():.2f} 米\n")

            # 如果有天空分割，计算地面区域统计
            if sky_masks is not None:
                sky_mask = sky_masks[i] > 0.5
                ground_depth = depth[~sky_mask]

                if len(ground_depth) > 0:
                    f.write(f"\n地面区域深度统计:\n")
                    f.write(f"  最小值: {ground_depth.min():.2f} 米\n")
                    f.write(f"  最大值: {ground_depth.max():.2f} 米\n")
                    f.write(f"  平均值: {ground_depth.mean():.2f} 米\n")
                    f.write(f"  中位数: {np.median(ground_depth):.2f} 米\n")
                    f.write(f"  标准差: {ground_depth.std():.2f} 米\n")

                    sky_ratio = sky_mask.sum() / sky_mask.size * 100
                    f.write(f"\n天空占比: {sky_ratio:.2f}%\n")

            # 深度分布（直方图）
            percentiles = [10, 25, 50, 75, 90]
            f.write(f"\n深度百分位数:\n")
            for p in percentiles:
                f.write(f"  {p}%: {np.percentile(depth, p):.2f} 米\n")

            f.write("\n\n")

    print(f"✓ 统计报告已保存: {save_path}")


def main():
    """主函数"""

    # ========== 配置 ==========
    print("=" * 70)
    print("使用真实标定文件进行 DA3 推理")
    print("=" * 70)

    # 数据路径（请修改为您的实际路径）
    DATA_DIR = Path("/home/zyc/路侧四个图")

    # 如果远程路径不存在，尝试相对路径
    if not DATA_DIR.exists():
        DATA_DIR = Path("./路侧四个图")

    if not DATA_DIR.exists():
        print(f"错误: 找不到数据目录 {DATA_DIR}")
        print("请修改脚本中的 DATA_DIR 路径")
        return

    # 相机参数文件（由 parse_calibration.py 生成）
    CAMERA_PARAMS_FILE = "./output/calibration/camera_params.npz"

    if not Path(CAMERA_PARAMS_FILE).exists():
        print(f"错误: 找不到相机参数文件 {CAMERA_PARAMS_FILE}")
        print("请先运行: python parse_calibration.py")
        return

    # 图像文件名模式
    IMAGE_PATTERN = "cam{}_1742877424148.png"
    CAMERA_IDS = ["0", "3", "6", "9"]

    # 模型路径
    MODEL_PATH = "/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"

    # 检查路径是否存在
    if not Path(MODEL_PATH).exists():
        print(f"警告: 模型路径不存在: {MODEL_PATH}")
        print(f"请检查模型文件位置或运行: python check_model_files.py")
        return

    # 输出目录
    OUTPUT_DIR = Path("./output/real_calibration_inference")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 推理配置
    UNDISTORT_IMAGES = False  # 是否对图像去畸变（您的图像已经去过畸变）
    PROCESS_RES = 504         # 处理分辨率

    # ========== 加载相机参数 ==========
    print("\n" + "-" * 70)
    print("加载相机参数")
    print("-" * 70)

    extrinsics, intrinsics, distortions, camera_names = load_camera_params(
        CAMERA_PARAMS_FILE
    )

    print(f"✓ 已加载 {len(extrinsics)} 个相机的参数")
    for i, name in enumerate(camera_names):
        print(f"  [{i}] {name}")

    # ========== 加载图像 ==========
    print("\n" + "-" * 70)
    print("加载图像")
    print("-" * 70)

    image_paths = []
    for cam_id in CAMERA_IDS:
        img_name = IMAGE_PATTERN.format(cam_id)
        img_path = DATA_DIR / img_name

        if not img_path.exists():
            print(f"错误: 找不到图像 {img_path}")
            return

        image_paths.append(img_path)
        print(f"✓ 找到图像: {img_name}")

    # 加载图像
    images = [Image.open(img_path) for img_path in image_paths]
    print(f"\n✓ 已加载 {len(images)} 张图像")
    print(f"  图像尺寸: {images[0].size}")

    # ========== 图像去畸变（可选） ==========
    if UNDISTORT_IMAGES:
        print("\n" + "-" * 70)
        print("图像去畸变处理")
        print("-" * 70)

        images_undistorted = undistort_images(images, intrinsics, distortions)

        # 保存去畸变后的图像
        undist_dir = OUTPUT_DIR / "undistorted_images"
        undist_dir.mkdir(exist_ok=True)

        for i, img in enumerate(images_undistorted):
            save_path = undist_dir / f"undistorted_{camera_names[i]}"
            img.save(save_path)
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
        print(f"✓ 模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("\n提示:")
        print("  1. 如果权重位置不对，参考 CAMERA_POSE_INFERENCE_GUIDE.md")
        print("  2. 或修改脚本中的 MODEL_PATH 为本地路径")
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
        # 先生成深度数据（NPZ格式）
        prediction = model.inference(
            image=images_for_inference,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=PROCESS_RES,
            align_to_input_ext_scale=True,  # 对齐到输入外参尺度
            export_dir=str(OUTPUT_DIR / "da3_output"),
            export_format="mini_npz",
        )

        print(f"\n✓ 深度推理完成!")
        print(f"  - 深度图 shape: {prediction.depth.shape}")
        print(f"  - 是否度量深度: {bool(prediction.is_metric)}")
        if prediction.sky is not None:
            print(f"  - 天空分割 shape: {prediction.sky.shape}")
        if prediction.conf is not None:
            print(f"  - 置信度图 shape: {prediction.conf.shape}")

        # 生成 3D 点云（GLB格式）
        print(f"\n正在生成 3D 点云...")
        prediction_pointcloud = model.inference(
            image=images_for_inference,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=PROCESS_RES,
            align_to_input_ext_scale=True,
            export_dir=str(OUTPUT_DIR / "pointcloud"),
            export_format="glb",
            conf_thresh_percentile=5.0,    # 降低置信度阈值（保留更多远距离点）
            num_max_points=5_000_000,      # 增加最大点数到500万
            show_cameras=True,             # 在点云中显示相机位置
        )

        print(f"  置信度阈值: 5% (保留95%的点)")
        print(f"  最大点数: 500万")

        print(f"\n✓ 3D 点云生成完成!")
        print(f"  - 点云文件: {OUTPUT_DIR / 'pointcloud' / 'prediction.glb'}")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 结果可视化 ==========
    print("\n" + "-" * 70)
    print("生成结果可视化")
    print("-" * 70)

    visualize_results(
        images=images,
        depth_maps=prediction.depth,
        sky_masks=prediction.sky,
        conf_maps=prediction.conf,
        camera_names=camera_names,
        save_dir=OUTPUT_DIR / "visualizations"
    )

    # ========== 统计分析 ==========
    print("\n" + "-" * 70)
    print("深度统计分析")
    print("-" * 70)

    for i in range(len(prediction.depth)):
        depth = prediction.depth[i]
        print(f"\n相机 {i} ({camera_names[i]}):")
        print(f"  深度范围: {depth.min():.2f} - {depth.max():.2f} 米")
        print(f"  平均深度: {depth.mean():.2f} 米")

        if prediction.sky is not None:
            sky_mask = prediction.sky[i] > 0.5
            ground_depth = depth[~sky_mask]
            if len(ground_depth) > 0:
                print(f"  地面平均深度: {ground_depth.mean():.2f} 米")

    # 生成详细报告
    report_path = OUTPUT_DIR / "depth_statistics_report.txt"
    create_depth_statistics_report(
        prediction.depth,
        prediction.sky,
        camera_names,
        report_path
    )

    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)

    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print(f"  - DA3 深度数据: {OUTPUT_DIR / 'da3_output' / 'prediction.npz'}")
    print(f"  - 3D 点云 (GLB): {OUTPUT_DIR / 'pointcloud' / 'prediction.glb'}")
    print(f"  - 结果可视化: {OUTPUT_DIR / 'visualizations' / 'result_*.png'}")
    print(f"  - 统计报告: {report_path}")
    if UNDISTORT_IMAGES:
        print(f"  - 去畸变图像: {OUTPUT_DIR / 'undistorted_images'}")

    print("\n提示:")
    print("  1. 使用 Blender/MeshLab/Online Viewer 打开 prediction.glb 查看 3D 点云")
    print("  2. 查看 visualizations/ 目录查看 2D 可视化结果")
    print("  3. 查看 depth_statistics_report.txt 了解详细统计")
    print("  4. 使用 prediction.npz 文件进行后续处理")
    print("\n在线查看器: https://gltf-viewer.donmccurdy.com/")


if __name__ == "__main__":
    main()
