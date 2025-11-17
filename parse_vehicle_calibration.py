#!/usr/bin/env python3
"""
解析车端周视相机标定文件 (NoEER705_v3)
将 YAML 格式的标定文件转换为 DA3 需要的相机参数格式
"""

import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(quat_xyzw):
    """
    将四元数转换为旋转矩阵

    Args:
        quat_xyzw: (x, y, z, w) 四元数

    Returns:
        R: (3, 3) 旋转矩阵
    """
    # scipy 使用 [x, y, z, w] 格式
    rot = Rotation.from_quat(quat_xyzw)
    return rot.as_matrix()


def load_camera_intrinsics(intrinsics_file):
    """
    加载相机内参

    Args:
        intrinsics_file: 内参文件路径

    Returns:
        K: (3, 3) 内参矩阵
        D: (5,) 畸变系数
        width: 图像宽度
        height: 图像高度
    """
    with open(intrinsics_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # K矩阵: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    K_list = data['K']
    K = np.array(K_list).reshape(3, 3)

    # 畸变系数: [k1, k2, p1, p2, k3]
    D = np.array(data['D'])

    # 图像尺寸
    width = data['width']
    height = data['height']

    return K, D, width, height


def load_camera_extrinsics(extrinsics_file):
    """
    加载相机外参

    Args:
        extrinsics_file: 外参文件路径

    Returns:
        ext: (4, 4) 外参矩阵 (lidar-to-camera, 即 world-to-camera)
    """
    with open(extrinsics_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    transform = data['transform']

    # 四元数 (x, y, z, w)
    quat = [
        transform['rotation']['x'],
        transform['rotation']['y'],
        transform['rotation']['z'],
        transform['rotation']['w']
    ]

    # 平移向量
    trans = np.array([
        transform['translation']['x'],
        transform['translation']['y'],
        transform['translation']['z']
    ])

    # 转换为旋转矩阵
    R = quaternion_to_rotation_matrix(quat)

    # 组合成4x4齐次变换矩阵 (camera-to-lidar, 根据label)
    c2l = np.eye(4)
    c2l[:3, :3] = R
    c2l[:3, 3] = trans

    # DA3 需要 lidar-to-camera (world-to-camera)，所以需要求逆
    ext = np.linalg.inv(c2l)

    return ext


def parse_vehicle_cameras(calib_dir, camera_ids):
    """
    解析车端周视相机的标定参数

    Args:
        calib_dir: NoEER705_v3 标定目录路径
        camera_ids: 要加载的相机ID列表，如 [1, 2, 3, 4, 5, 6, 7]

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变参数
        camera_names: 相机名称列表
        image_sizes: [(width, height), ...] 图像尺寸列表
    """
    calib_dir = Path(calib_dir)
    camera_dir = calib_dir / 'camera'

    extrinsics = []
    intrinsics = []
    distortions = []
    camera_names = []
    image_sizes = []

    # 相机名称映射
    camera_info = {
        1: '前左下 (FN)',
        2: '前右下 (FW)',
        3: '前左上 (FL)',
        4: '前右上 (FR)',
        5: '左后下 (RL)',
        6: '右后下 (RR)',
        7: '后中下 (RN)',
    }

    print("=" * 70)
    print("解析车端周视相机标定文件")
    print("=" * 70)

    for cam_id in camera_ids:
        camera_name = f"camera_{cam_id:02d}"
        info_str = camera_info.get(cam_id, f'相机{cam_id}')

        # 文件路径
        intrinsics_file = camera_dir / f"{camera_name}_intrinsics.yaml"
        extrinsics_file = camera_dir / f"{camera_name}_extrinsics.yaml"

        if not intrinsics_file.exists():
            print(f"警告: 找不到内参文件 {intrinsics_file}")
            continue
        if not extrinsics_file.exists():
            print(f"警告: 找不到外参文件 {extrinsics_file}")
            continue

        # 加载参数
        K, D, width, height = load_camera_intrinsics(intrinsics_file)
        ext = load_camera_extrinsics(extrinsics_file)

        extrinsics.append(ext)
        intrinsics.append(K)
        distortions.append(D)
        camera_names.append(camera_name)
        image_sizes.append((width, height))

        print(f"\n相机 {cam_id} - {info_str}:")
        print(f"  文件: {camera_name}")
        print(f"  图像尺寸: {width} x {height}")
        print(f"  焦距 (fx, fy): ({K[0,0]:.2f}, {K[1,1]:.2f})")
        print(f"  主点 (cx, cy): ({K[0,2]:.2f}, {K[1,2]:.2f})")
        print(f"  畸变系数: {D}")
        print(f"  位置 (x, y, z): ({ext[0,3]:.3f}, {ext[1,3]:.3f}, {ext[2,3]:.3f})")

    return (
        np.stack(extrinsics),
        np.stack(intrinsics),
        np.stack(distortions),
        camera_names,
        image_sizes
    )


def save_camera_params(save_path, extrinsics, intrinsics, distortions, camera_names, image_sizes):
    """
    保存相机参数为 .npz 文件

    Args:
        save_path: 保存路径
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变参数
        camera_names: 相机名称列表
        image_sizes: 图像尺寸列表
    """
    np.savez(
        save_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortions=distortions,
        camera_names=camera_names,
        image_sizes=image_sizes
    )
    print(f"\n相机参数已保存到: {save_path}")


def main():
    """主函数 - 示例用法"""

    # ========== 配置 ==========
    # 标定目录（请修改为您的实际路径）
    CALIB_DIR = Path("./NoEER705_v3")

    # 要使用的相机ID（1-7对应前视、左右、后视）
    CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7]

    # 输出目录
    OUTPUT_DIR = Path("./output/vehicle_surround_calibration")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== 解析标定文件 ==========
    extrinsics, intrinsics, distortions, camera_names, image_sizes = parse_vehicle_cameras(
        CALIB_DIR,
        CAMERA_IDS
    )

    # ========== 保存参数 ==========
    save_path = OUTPUT_DIR / "vehicle_camera_params.npz"
    save_camera_params(save_path, extrinsics, intrinsics, distortions, camera_names, image_sizes)

    # ========== 验证加载 ==========
    print("\n" + "=" * 70)
    print("验证保存的参数")
    print("=" * 70)

    data = np.load(save_path, allow_pickle=True)
    print(f"✓ 成功加载保存的参数")
    print(f"  - extrinsics shape: {data['extrinsics'].shape}")
    print(f"  - intrinsics shape: {data['intrinsics'].shape}")
    print(f"  - distortions shape: {data['distortions'].shape}")
    print(f"  - camera_names: {list(data['camera_names'])}")
    print(f"  - image_sizes: {list(data['image_sizes'])}")

    # ========== 导出为文本格式 ==========
    txt_path = OUTPUT_DIR / "vehicle_camera_params.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("车端周视相机参数\n")
        f.write("=" * 70 + "\n\n")

        for i in range(len(extrinsics)):
            f.write(f"相机 {i+1} ({camera_names[i]}):\n")
            f.write("-" * 70 + "\n")

            f.write(f"图像尺寸: {image_sizes[i][0]} x {image_sizes[i][1]}\n\n")

            f.write("内参矩阵 (3x3):\n")
            f.write(str(intrinsics[i]) + "\n\n")

            f.write("外参矩阵 (4x4, lidar-to-camera):\n")
            f.write(str(extrinsics[i]) + "\n\n")

            f.write("畸变系数 [k1, k2, p1, p2, k3]:\n")
            f.write(str(distortions[i]) + "\n\n\n")

    print(f"\n参数文本文件已保存到: {txt_path}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n现在可以使用这些参数进行 DA3 推理：")
    print(f"  python inference_vehicle_surround.py")


if __name__ == "__main__":
    main()
