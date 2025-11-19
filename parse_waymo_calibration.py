#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo 数据集标定文件解析工具

支持解析 Waymo 格式的相机内外参：
- 外参：4x4 变换矩阵（TXT文件）
- 内参：fx, fy, cx, cy + 畸变系数（TXT文件）
"""

import numpy as np
from pathlib import Path


def load_waymo_intrinsics(intrinsics_file):
    """
    加载 Waymo 相机内参

    Args:
        intrinsics_file: 内参文件路径

    Returns:
        K: (3, 3) 内参矩阵
        D: (5,) 畸变系数 [k1, k2, p1, p2, k3]
    """
    with open(intrinsics_file, 'r') as f:
        lines = f.readlines()

    # 解析参数（每行一个数字）
    params = [float(line.strip()) for line in lines]

    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]
    k1 = params[4]
    k2 = params[5]
    p1 = params[6]
    p2 = params[7]
    k3 = params[8] if len(params) > 8 else 0.0

    # 构建内参矩阵
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 畸变系数
    D = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    return K, D


def load_waymo_extrinsics(extrinsics_file):
    """
    加载 Waymo 相机外参

    Args:
        extrinsics_file: 外参文件路径

    Returns:
        ext: (4, 4) 外参矩阵

    Note:
        Waymo 外参通常是 camera-to-world (c2w) 变换
        DA3 需要 world-to-camera (w2c)，所以需要求逆
    """
    # 读取4x4矩阵
    ext = np.loadtxt(extrinsics_file, dtype=np.float32)

    # 验证格式
    if ext.shape != (4, 4):
        raise ValueError(f"外参矩阵应为 4x4，实际为 {ext.shape}")

    # Waymo 通常给的是 camera-to-world，需要求逆得到 world-to-camera
    # DA3 需要 world-to-camera (w2c)
    w2c = np.linalg.inv(ext)

    return w2c


def parse_waymo_calibration(calib_dir, camera_ids):
    """
    批量解析 Waymo 相机标定

    Args:
        calib_dir: 标定文件根目录（包含 intrinsics/ 和 extrinsics/ 子目录）
        camera_ids: 相机ID列表，例如 [0, 1, 2, 3, 4]

    Returns:
        intrinsics: List of (3, 3) 内参矩阵
        extrinsics: List of (4, 4) 外参矩阵（world-to-camera）
        distortions: List of (5,) 畸变系数
        camera_names: List of str，相机名称
    """
    calib_dir = Path(calib_dir)
    intrinsics_dir = calib_dir / "intrinsics"
    extrinsics_dir = calib_dir / "extrinsics"

    intrinsics = []
    extrinsics = []
    distortions = []
    camera_names = []

    print("=" * 70)
    print("解析 Waymo 相机标定")
    print("=" * 70)

    for cam_id in camera_ids:
        intri_file = intrinsics_dir / f"{cam_id}.txt"
        extri_file = extrinsics_dir / f"{cam_id}.txt"

        if not intri_file.exists():
            raise FileNotFoundError(f"内参文件不存在: {intri_file}")
        if not extri_file.exists():
            raise FileNotFoundError(f"外参文件不存在: {extri_file}")

        # 加载内参和畸变
        K, D = load_waymo_intrinsics(intri_file)
        intrinsics.append(K)
        distortions.append(D)

        # 加载外参
        ext = load_waymo_extrinsics(extri_file)
        extrinsics.append(ext)

        # 相机名称
        cam_name = f"camera_{cam_id}"
        camera_names.append(cam_name)

        # 打印信息
        print(f"\n相机 {cam_id}:")
        print(f"  - 内参 (fx, fy, cx, cy): "
              f"({K[0,0]:.2f}, {K[1,1]:.2f}, {K[0,2]:.2f}, {K[1,2]:.2f})")
        print(f"  - 畸变系数 [k1, k2, p1, p2, k3]: {D}")
        print(f"  - 位置 (x, y, z): ({ext[0,3]:.2f}, {ext[1,3]:.2f}, {ext[2,3]:.2f})")

    print("\n" + "=" * 70)
    print(f"成功解析 {len(camera_ids)} 个相机的标定参数")
    print("=" * 70)

    return intrinsics, extrinsics, distortions, camera_names


def save_waymo_calibration(intrinsics, extrinsics, distortions, camera_names, output_dir):
    """
    保存解析后的标定参数到 NPZ 文件

    Args:
        intrinsics: List of (3, 3) 内参矩阵
        extrinsics: List of (4, 4) 外参矩阵
        distortions: List of (5,) 畸变系数
        camera_names: List of str，相机名称
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "waymo_camera_params.npz"

    np.savez(
        output_file,
        intrinsics=np.array(intrinsics),
        extrinsics=np.array(extrinsics),
        distortions=np.array(distortions),
        camera_names=np.array(camera_names)
    )

    print(f"\n标定参数已保存到: {output_file}")
    print(f"  - 相机数量: {len(camera_names)}")
    print(f"  - 包含: 内参、外参、畸变系数、相机名称")


if __name__ == "__main__":
    # 示例用法
    CALIB_DIR = "/home/zyc/street_crafter/data/waymo/049"
    CAMERA_IDS = [0, 1, 2, 3, 4]  # Waymo 有5个相机
    OUTPUT_DIR = "output/waymo_calibration"

    # 解析标定
    intrinsics, extrinsics, distortions, camera_names = parse_waymo_calibration(
        CALIB_DIR, CAMERA_IDS
    )

    # 保存到文件
    save_waymo_calibration(intrinsics, extrinsics, distortions, camera_names, OUTPUT_DIR)
