#!/usr/bin/env python3
"""
DA3 推理脚本 - 带有相机位姿
使用 Depth Anything V3 模型进行深度估计，支持相机位姿输入和输出
"""

import torch
import numpy as np
from pathlib import Path
from depth_anything_3.api import DepthAnything3
from PIL import Image


def create_sample_camera_params(num_images, image_width=1920, image_height=1080):
    """
    创建示例相机参数

    Args:
        num_images: 图像数量
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵 (world-to-camera)
        intrinsics: (N, 3, 3) 内参矩阵
    """
    extrinsics = []
    intrinsics = []

    for i in range(num_images):
        # 外参：世界到相机的变换矩阵 [R | t]
        # 这里创建一个简单的相机序列（沿X轴移动）
        ext = np.eye(4)
        ext[0, 3] = i * 1.0  # X方向平移
        extrinsics.append(ext)

        # 内参：相机矩阵
        # [[fx, 0, cx],
        #  [0, fy, cy],
        #  [0,  0,  1]]
        fx = fy = 1000.0  # 焦距
        cx = image_width / 2.0
        cy = image_height / 2.0

        intr = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics.append(intr)

    return np.stack(extrinsics), np.stack(intrinsics)


def load_camera_params_from_file(camera_file, num_images):
    """
    从文件加载相机参数

    Args:
        camera_file: 相机参数文件路径 (.npz 格式)
        num_images: 期望的图像数量

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
    """
    data = np.load(camera_file)

    extrinsics = data['extrinsics']  # (N, 4, 4)
    intrinsics = data['intrinsics']  # (N, 3, 3)

    assert extrinsics.shape[0] == num_images, f"相机参数数量不匹配: {extrinsics.shape[0]} != {num_images}"

    return extrinsics, intrinsics


def inference_with_camera_pose(
    model_path: str,
    image_paths: list[str],
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    output_dir: str = "./output",
    device: str = "cuda"
):
    """
    使用相机位姿进行推理

    Args:
        model_path: 模型权重路径或 Hugging Face 模型名称
        image_paths: 图像路径列表
        extrinsics: (N, 4, 4) 外参矩阵，如果为 None 则模型会估计
        intrinsics: (N, 3, 3) 内参矩阵，如果为 None 则模型会估计
        output_dir: 输出目录
        device: 计算设备 ('cuda' 或 'cpu')

    Returns:
        prediction: 预测结果对象
    """
    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 如果是本地路径，使用 from_pretrained 加载
    if Path(model_path).exists():
        model = DepthAnything3.from_pretrained(model_path)
    else:
        # 否则从 Hugging Face 加载
        model = DepthAnything3.from_pretrained(model_path)

    model = model.to(device=device)
    print(f"模型已加载到 {device}")

    # 2. 加载图像
    print(f"正在加载 {len(image_paths)} 张图像...")
    images = [Image.open(img_path) for img_path in image_paths]

    # 3. 如果没有提供相机参数，创建示例参数
    if extrinsics is None or intrinsics is None:
        print("未提供相机参数，将由模型自动估计")
        extrinsics = None
        intrinsics = None
    else:
        print(f"使用提供的相机参数: extrinsics shape={extrinsics.shape}, intrinsics shape={intrinsics.shape}")

    # 4. 执行推理
    print("正在执行推理...")
    prediction = model.inference(
        image=images,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=504,  # 处理分辨率
        export_dir=output_dir,
        export_format="mini_npz",  # 导出格式：mini_npz, npz, glb, gs_ply
        infer_metric=True,  # 推理度量深度
        infer_sky=True,  # 推理天空分割
        infer_gs=False,  # 是否推理高斯点云（需要 da3-giant 模型）
    )

    # 5. 输出结果信息
    print("\n推理完成！结果信息:")
    print(f"  - 深度图 shape: {prediction.depth.shape}")
    print(f"  - 是否为度量深度: {bool(prediction.is_metric)}")

    if prediction.extrinsics is not None:
        print(f"  - 估计的外参 shape: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"  - 估计的内参 shape: {prediction.intrinsics.shape}")
    if prediction.sky is not None:
        print(f"  - 天空分割 shape: {prediction.sky.shape}")
    if prediction.conf is not None:
        print(f"  - 置信度图 shape: {prediction.conf.shape}")

    print(f"\n结果已保存到: {output_dir}")

    return prediction


def main():
    """主函数 - 示例用法"""

    # ========== 配置参数 ==========

    # 模型路径（请修改为您的实际路径）
    # 方式1: 使用 Hugging Face 模型名称（会自动下载）
    MODEL_PATH = "depth-anything/DA3NESTED-GIANT-LARGE"

    # 方式2: 使用本地路径（您下载的权重）
    # MODEL_PATH = "/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1"
    # 注意：建议将权重移动到正确的位置：
    # mkdir -p /home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE
    # mv /home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1/* \
    #    /home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/

    # 图像路径（请修改为您的实际图像）
    IMAGE_DIR = Path("./assets/examples/SOH")  # 示例图像目录
    image_paths = sorted(IMAGE_DIR.glob("*.png"))[:4]  # 取前4张图像

    # 输出目录
    OUTPUT_DIR = "./output/with_camera_pose"

    # ========== 示例1: 不提供相机参数，让模型自动估计 ==========
    print("=" * 60)
    print("示例1: 自动估计相机位姿")
    print("=" * 60)

    prediction1 = inference_with_camera_pose(
        model_path=MODEL_PATH,
        image_paths=image_paths,
        extrinsics=None,  # 不提供，让模型估计
        intrinsics=None,  # 不提供，让模型估计
        output_dir=OUTPUT_DIR + "/auto_estimate",
        device="cuda"
    )

    # ========== 示例2: 提供相机参数进行推理 ==========
    print("\n" + "=" * 60)
    print("示例2: 使用提供的相机参数")
    print("=" * 60)

    # 创建示例相机参数
    num_images = len(image_paths)
    extrinsics, intrinsics = create_sample_camera_params(num_images)

    prediction2 = inference_with_camera_pose(
        model_path=MODEL_PATH,
        image_paths=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_dir=OUTPUT_DIR + "/with_camera_params",
        device="cuda"
    )

    # ========== 示例3: 从文件加载相机参数 ==========
    print("\n" + "=" * 60)
    print("示例3: 从文件加载相机参数")
    print("=" * 60)

    # 首先保存一个示例相机参数文件
    camera_params_file = Path(OUTPUT_DIR) / "camera_params.npz"
    camera_params_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        camera_params_file,
        extrinsics=extrinsics,
        intrinsics=intrinsics
    )
    print(f"示例相机参数已保存到: {camera_params_file}")

    # 从文件加载
    extrinsics_loaded, intrinsics_loaded = load_camera_params_from_file(
        camera_params_file, num_images
    )

    prediction3 = inference_with_camera_pose(
        model_path=MODEL_PATH,
        image_paths=image_paths,
        extrinsics=extrinsics_loaded,
        intrinsics=intrinsics_loaded,
        output_dir=OUTPUT_DIR + "/from_file",
        device="cuda"
    )

    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
