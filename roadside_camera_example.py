#!/usr/bin/env python3
"""
路侧相机深度估计示例
针对路侧监控场景的 DA3 推理脚本
"""

import torch
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from PIL import Image
import matplotlib.pyplot as plt


def create_roadside_camera_params(
    num_cameras=4,
    camera_height=5.0,
    camera_spacing=10.0,
    pitch_angle=30.0,
    image_width=1920,
    image_height=1080,
    focal_length=1500.0
):
    """
    创建路侧相机的相机参数

    Args:
        num_cameras: 相机数量
        camera_height: 相机安装高度（米）
        camera_spacing: 相机沿道路方向的间距（米）
        pitch_angle: 相机俯仰角（度）
        image_width: 图像宽度
        image_height: 图像高度
        focal_length: 焦距（像素）

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
    """
    extrinsics = []
    intrinsics = []

    for i in range(num_cameras):
        # ========== 外参矩阵 ==========
        ext = np.eye(4)

        # 1. 设置相机位置
        # 假设道路沿 X 轴，相机沿道路等间距分布
        ext[0, 3] = i * camera_spacing  # X: 沿道路方向
        ext[1, 3] = camera_height       # Y: 安装高度
        ext[2, 3] = 0.0                 # Z: 道路中心线偏移

        # 2. 设置相机旋转（俯视角度）
        # 绕 X 轴旋转（俯仰角）
        pitch_rad = np.radians(pitch_angle)
        rot_x = np.eye(4)
        rot_x[1, 1] = np.cos(pitch_rad)
        rot_x[1, 2] = -np.sin(pitch_rad)
        rot_x[2, 1] = np.sin(pitch_rad)
        rot_x[2, 2] = np.cos(pitch_rad)

        # 应用旋转
        ext = ext @ rot_x

        extrinsics.append(ext)

        # ========== 内参矩阵 ==========
        # 路侧监控相机通常使用较大焦距（长焦）
        fx = fy = focal_length
        cx = image_width / 2.0
        cy = image_height / 2.0

        intr = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        intrinsics.append(intr)

    return np.stack(extrinsics), np.stack(intrinsics)


def visualize_depth_with_stats(depth, save_path, title="Depth Map"):
    """
    可视化深度图并显示统计信息

    Args:
        depth: (H, W) 深度图
        save_path: 保存路径
        title: 标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 深度图
    im = ax1.imshow(depth, cmap='turbo')
    ax1.set_title(title)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 深度直方图
    ax2.hist(depth.flatten(), bins=100, color='skyblue', edgecolor='black')
    ax2.set_title('Depth Distribution')
    ax2.set_xlabel('Depth (m)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"Min: {depth.min():.2f}m\n"
    stats_text += f"Max: {depth.max():.2f}m\n"
    stats_text += f"Mean: {depth.mean():.2f}m\n"
    stats_text += f"Median: {np.median(depth):.2f}m"
    ax2.text(0.95, 0.95, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化已保存到: {save_path}")


def main():
    """主函数"""

    # ========== 配置 ==========
    print("=" * 70)
    print("路侧相机深度估计")
    print("=" * 70)

    # 图像路径（请修改为您的实际路径）
    IMAGE_DIR = Path("./路侧四个图")  # 您的图像目录

    # 如果目录不存在，使用示例图像
    if not IMAGE_DIR.exists():
        print(f"警告: 目录 {IMAGE_DIR} 不存在，使用默认示例目录")
        IMAGE_DIR = Path("./assets/examples/SOH")

    # 加载图像
    image_paths = sorted(IMAGE_DIR.glob("*.png"))
    if not image_paths:
        image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        print(f"错误: 在 {IMAGE_DIR} 中没有找到图像文件")
        return

    # 限制图像数量（如果太多）
    MAX_IMAGES = 4
    if len(image_paths) > MAX_IMAGES:
        print(f"发现 {len(image_paths)} 张图像，仅处理前 {MAX_IMAGES} 张")
        image_paths = image_paths[:MAX_IMAGES]

    print(f"\n找到 {len(image_paths)} 张图像:")
    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}] {img_path.name}")

    # 模型路径
    MODEL_PATH = "depth-anything/DA3NESTED-GIANT-LARGE"
    # 或使用本地路径:
    # MODEL_PATH = "/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1"

    # 输出目录
    OUTPUT_DIR = Path("./output/roadside_cameras")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== 创建路侧相机参数 ==========
    print("\n" + "-" * 70)
    print("创建路侧相机参数")
    print("-" * 70)

    # 路侧相机配置
    CAMERA_CONFIG = {
        "num_cameras": len(image_paths),
        "camera_height": 5.0,      # 相机高度 5 米
        "camera_spacing": 15.0,    # 相机间距 15 米
        "pitch_angle": 25.0,       # 俯视角度 25 度
        "focal_length": 1200.0,    # 焦距（像素）
    }

    print(f"相机配置:")
    print(f"  - 数量: {CAMERA_CONFIG['num_cameras']}")
    print(f"  - 安装高度: {CAMERA_CONFIG['camera_height']} 米")
    print(f"  - 间距: {CAMERA_CONFIG['camera_spacing']} 米")
    print(f"  - 俯仰角: {CAMERA_CONFIG['pitch_angle']} 度")
    print(f"  - 焦距: {CAMERA_CONFIG['focal_length']} 像素")

    extrinsics, intrinsics = create_roadside_camera_params(**CAMERA_CONFIG)

    # 保存相机参数
    camera_params_file = OUTPUT_DIR / "camera_params.npz"
    np.savez(
        camera_params_file,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        config=CAMERA_CONFIG
    )
    print(f"\n相机参数已保存到: {camera_params_file}")

    # ========== 加载模型 ==========
    print("\n" + "-" * 70)
    print("加载 DA3 模型")
    print("-" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        model = DepthAnything3.from_pretrained(MODEL_PATH)
        model = model.to(device=device)
        print(f"模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n提示: 如果是本地路径错误，请参考 CAMERA_POSE_INFERENCE_GUIDE.md")
        return

    # ========== 执行推理 ==========
    print("\n" + "-" * 70)
    print("执行推理")
    print("-" * 70)

    # 方式1: 使用相机参数
    print("\n[方式1] 使用提供的路侧相机参数")
    prediction_with_params = model.inference(
        image=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=504,
        infer_metric=True,         # 度量深度（真实尺度）
        infer_sky=True,            # 天空分割
        export_dir=str(OUTPUT_DIR / "with_camera_params"),
        export_format="mini_npz",
        export_depth_vis=True,     # 自动导出深度可视化
        export_sky_vis=True,       # 自动导出天空可视化
    )

    print(f"✓ 推理完成")
    print(f"  - 深度图 shape: {prediction_with_params.depth.shape}")
    print(f"  - 度量深度: {bool(prediction_with_params.is_metric)}")
    if prediction_with_params.sky is not None:
        print(f"  - 天空分割 shape: {prediction_with_params.sky.shape}")

    # 方式2: 不使用相机参数（让模型自动估计）
    print("\n[方式2] 让模型自动估计相机参数")
    prediction_auto = model.inference(
        image=image_paths,
        extrinsics=None,           # 不提供
        intrinsics=None,           # 不提供
        process_res=504,
        infer_metric=True,
        export_dir=str(OUTPUT_DIR / "auto_estimate"),
        export_format="mini_npz",
        export_depth_vis=True,
    )

    print(f"✓ 推理完成")
    print(f"  - 深度图 shape: {prediction_auto.depth.shape}")
    if prediction_auto.extrinsics is not None:
        print(f"  - 估计的外参 shape: {prediction_auto.extrinsics.shape}")
    if prediction_auto.intrinsics is not None:
        print(f"  - 估计的内参 shape: {prediction_auto.intrinsics.shape}")

    # ========== 深度分析 ==========
    print("\n" + "-" * 70)
    print("深度统计分析")
    print("-" * 70)

    for i in range(len(image_paths)):
        print(f"\n图像 {i+1}: {image_paths[i].name}")

        depth = prediction_with_params.depth[i]

        print(f"  深度范围: {depth.min():.2f} - {depth.max():.2f} 米")
        print(f"  平均深度: {depth.mean():.2f} 米")
        print(f"  中位数深度: {np.median(depth):.2f} 米")

        # 如果有天空分割，计算地面区域的深度
        if prediction_with_params.sky is not None:
            sky_mask = prediction_with_params.sky[i] > 0.5
            ground_depth = depth[~sky_mask]
            if len(ground_depth) > 0:
                print(f"  地面平均深度: {ground_depth.mean():.2f} 米")
                print(f"  地面深度范围: {ground_depth.min():.2f} - {ground_depth.max():.2f} 米")

        # 创建详细可视化
        vis_path = OUTPUT_DIR / f"depth_analysis_{i:03d}.png"
        visualize_depth_with_stats(
            depth,
            vis_path,
            title=f"Depth Map - {image_paths[i].name}"
        )

    # ========== 完成 ==========
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print(f"  - 相机参数: {camera_params_file}")
    print(f"  - 深度预测: {OUTPUT_DIR / 'with_camera_params' / 'prediction.npz'}")
    print(f"  - 自动估计: {OUTPUT_DIR / 'auto_estimate' / 'prediction.npz'}")
    print(f"  - 深度分析: {OUTPUT_DIR / 'depth_analysis_*.png'}")
    print(f"  - 深度可视化: {OUTPUT_DIR / 'with_camera_params' / 'depth_vis_*.png'}")

    print("\n提示:")
    print("  1. 查看 CAMERA_POSE_INFERENCE_GUIDE.md 了解详细文档")
    print("  2. 修改 CAMERA_CONFIG 调整相机参数")
    print("  3. 使用 .npz 文件加载深度数据进行后续处理")


if __name__ == "__main__":
    main()
