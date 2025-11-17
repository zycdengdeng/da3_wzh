#!/usr/bin/env python3
"""
解析路侧相机标定文件
将标定文件转换为 DA3 需要的相机参数格式
"""

import json
import numpy as np
import cv2
from pathlib import Path


def rodrigues_to_rotation_matrix(rvec):
    """
    将罗德里格斯向量转换为旋转矩阵

    Args:
        rvec: (3,) 罗德里格斯向量

    Returns:
        R: (3, 3) 旋转矩阵
    """
    R, _ = cv2.Rodrigues(np.array(rvec))
    return R


def parse_camera_params(calib_data, camera_ids):
    """
    解析标定文件中的相机参数

    Args:
        calib_data: 标定文件的字典数据
        camera_ids: 要提取的相机ID列表，如 ["0", "3", "6", "9"]

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵 (virtualLidar-to-camera)
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变参数
        camera_names: 相机名称列表
    """
    extrinsics = []
    intrinsics = []
    distortions = []
    camera_names = []

    for cam_id in camera_ids:
        if cam_id not in calib_data["camera"]:
            raise ValueError(f"相机 {cam_id} 不在标定文件中")

        cam_data = calib_data["camera"][cam_id]

        # 1. 内参矩阵 (3x3)
        intri = np.array(cam_data["intri"]).reshape(3, 3)
        intrinsics.append(intri)

        # 2. 畸变参数
        distor = np.array(cam_data["distor"])
        distortions.append(distor)

        # 3. 外参矩阵：virtualLidarToCam
        # 注意：这是 virtualLidar 坐标系到 camera 坐标系的变换
        vl2cam = cam_data["virtualLidarToCam"]

        # 将罗德里格斯向量转换为旋转矩阵
        rvec = np.array(vl2cam["rotate"])
        R = rodrigues_to_rotation_matrix(rvec)

        # 平移向量
        t = np.array(vl2cam["trans"]).reshape(3, 1)

        # 组合成 4x4 齐次变换矩阵
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = t.squeeze()

        extrinsics.append(ext)

        # 相机名称
        camera_names.append(cam_data["name"])

        # 打印信息
        print(f"\n相机 {cam_id} ({cam_data['name']}):")
        print(f"  - 鱼眼: {'是' if cam_data['isFish'] else '否'}")
        print(f"  - 内参 (fx, fy, cx, cy): "
              f"({intri[0,0]:.2f}, {intri[1,1]:.2f}, {intri[0,2]:.2f}, {intri[1,2]:.2f})")
        print(f"  - 畸变系数: {distor}")
        print(f"  - 位置 (x, y, z): ({t[0,0]:.2f}, {t[1,0]:.2f}, {t[2,0]:.2f})")

    return (
        np.stack(extrinsics),
        np.stack(intrinsics),
        np.stack(distortions),
        camera_names
    )


def load_calibration_file(calib_file, camera_ids=["0", "3", "6", "9"]):
    """
    加载标定文件并解析相机参数

    Args:
        calib_file: 标定文件路径 (.json)
        camera_ids: 要加载的相机ID列表

    Returns:
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变参数
        camera_names: 相机名称列表
    """
    print(f"正在加载标定文件: {calib_file}")

    with open(calib_file, 'r') as f:
        calib_data = json.load(f)

    print(f"数据集: {calib_data['data']}")
    print(f"图像尺寸 (非鱼眼): {calib_data['imgSize']['notFish']}")
    print(f"可用相机: {list(calib_data['camera'].keys())}")
    print(f"选择相机: {camera_ids}")

    extrinsics, intrinsics, distortions, camera_names = parse_camera_params(
        calib_data, camera_ids
    )

    return extrinsics, intrinsics, distortions, camera_names


def save_camera_params(save_path, extrinsics, intrinsics, distortions, camera_names):
    """
    保存相机参数为 .npz 文件

    Args:
        save_path: 保存路径
        extrinsics: (N, 4, 4) 外参矩阵
        intrinsics: (N, 3, 3) 内参矩阵
        distortions: (N, 5) 畸变参数
        camera_names: 相机名称列表
    """
    np.savez(
        save_path,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortions=distortions,
        camera_names=camera_names
    )
    print(f"\n相机参数已保存到: {save_path}")


def visualize_camera_setup(extrinsics, camera_names):
    """
    打印相机布局信息

    Args:
        extrinsics: (N, 4, 4) 外参矩阵
        camera_names: 相机名称列表
    """
    print("\n" + "=" * 70)
    print("相机空间布局（相对于虚拟雷达坐标系）")
    print("=" * 70)

    for i, (ext, name) in enumerate(zip(extrinsics, camera_names)):
        pos = ext[:3, 3]
        print(f"\n相机 {i} ({name}):")
        print(f"  位置 (x, y, z): ({pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}) 米")

        # 提取旋转角度（欧拉角）
        R = ext[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        print(f"  旋转向量: ({rvec[0,0]:6.3f}, {rvec[1,0]:6.3f}, {rvec[2,0]:6.3f})")

    # 计算相机之间的距离
    print("\n" + "-" * 70)
    print("相机间距")
    print("-" * 70)
    for i in range(len(extrinsics)):
        for j in range(i+1, len(extrinsics)):
            pos_i = extrinsics[i][:3, 3]
            pos_j = extrinsics[j][:3, 3]
            dist = np.linalg.norm(pos_i - pos_j)
            print(f"  相机 {i} <-> 相机 {j}: {dist:8.2f} 米")


def main():
    """主函数 - 示例用法"""

    # ========== 配置 ==========
    # 标定文件路径（请修改为您的实际路径）
    CALIB_FILE = "/home/zyc/路侧四个图/calib.json"

    # 如果找不到，尝试相对路径
    if not Path(CALIB_FILE).exists():
        CALIB_FILE = "./路侧四个图/calib.json"

    if not Path(CALIB_FILE).exists():
        print(f"错误: 找不到标定文件 {CALIB_FILE}")
        print("请修改脚本中的 CALIB_FILE 路径")
        return

    # 要使用的相机ID
    CAMERA_IDS = ["0", "3", "6", "9"]

    # 输出路径
    OUTPUT_DIR = Path("./output/calibration")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== 加载和解析标定文件 ==========
    print("=" * 70)
    print("解析相机标定文件")
    print("=" * 70)

    extrinsics, intrinsics, distortions, camera_names = load_calibration_file(
        CALIB_FILE,
        CAMERA_IDS
    )

    # ========== 可视化相机布局 ==========
    visualize_camera_setup(extrinsics, camera_names)

    # ========== 保存参数 ==========
    save_path = OUTPUT_DIR / "camera_params.npz"
    save_camera_params(save_path, extrinsics, intrinsics, distortions, camera_names)

    # ========== 验证加载 ==========
    print("\n" + "=" * 70)
    print("验证保存的参数")
    print("=" * 70)

    data = np.load(save_path)
    print(f"✓ 成功加载保存的参数")
    print(f"  - extrinsics shape: {data['extrinsics'].shape}")
    print(f"  - intrinsics shape: {data['intrinsics'].shape}")
    print(f"  - distortions shape: {data['distortions'].shape}")
    print(f"  - camera_names: {list(data['camera_names'])}")

    # ========== 导出为文本格式（便于检查） ==========
    txt_path = OUTPUT_DIR / "camera_params.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("相机参数\n")
        f.write("=" * 70 + "\n\n")

        for i in range(len(extrinsics)):
            f.write(f"相机 {i} ({camera_names[i]}):\n")
            f.write("-" * 70 + "\n")

            f.write("内参矩阵 (3x3):\n")
            f.write(str(intrinsics[i]) + "\n\n")

            f.write("外参矩阵 (4x4):\n")
            f.write(str(extrinsics[i]) + "\n\n")

            f.write("畸变系数:\n")
            f.write(str(distortions[i]) + "\n\n")

    print(f"\n参数文本文件已保存到: {txt_path}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n现在可以使用这些参数进行 DA3 推理：")
    print(f"  python inference_with_real_calibration.py")


if __name__ == "__main__":
    main()
