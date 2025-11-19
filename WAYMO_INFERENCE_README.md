# Waymo 数据集 DA3 推理指南

本指南介绍如何使用 Depth Anything V3 (DA3) 对 Waymo 数据集进行批量深度推理。

---

## ⚡ 环境要求

- **Python 环境**: 需要已安装 `depth_anything_3` 包的 conda/venv 环境
- **必需依赖**: numpy, opencv-python, pillow
- **DA3 安装**: `depth_anything_3` 应通过 pip 安装在环境中

**验证安装**:
```bash
python -c "from depth_anything_3.api import DepthAnything3; print('✓ DA3 已安装')"
```

---

## 📁 数据格式

### 目录结构

```
/home/zyc/street_crafter/data/waymo/049/
├── intrinsics/          # 内参文件
│   ├── 0.txt
│   ├── 1.txt
│   ├── 2.txt
│   ├── 3.txt
│   └── 4.txt
├── extrinsics/          # 外参文件
│   ├── 0.txt
│   ├── 1.txt
│   ├── 2.txt
│   ├── 3.txt
│   └── 4.txt
└── images/              # 图像
    ├── 000000_0.png     # 帧0，相机0
    ├── 000000_1.png     # 帧0，相机1
    ├── 000000_2.png
    ├── 000000_3.png
    ├── 000000_4.png
    ├── 000001_0.png     # 帧1，相机0
    └── ...
```

### 内参文件格式 (`intrinsics/X.txt`)

9行文本，每行一个数字：

```
2.084604312956008926e+03    # fx
2.084604312956008926e+03    # fy
9.334067577078354816e+02    # cx
6.650223418347507049e+02    # cy
4.703917157737932259e-02    # k1
-3.573563265746588224e-01   # k2
2.367692675182160307e-04    # p1
-4.036682712118284250e-04   # p2
0.000000000000000000e+00    # k3
```

### 外参文件格式 (`extrinsics/X.txt`)

4x4 齐次变换矩阵（camera-to-world）：

```
-4.588266398430291063e-03 -3.413667297520365119e-03 9.999836472098125872e-01 1.544154267170511075e+00
-9.999632354307952387e-01 -7.228375769820964344e-03 -4.612848415714690224e-03 -2.315740942895095494e-02
7.244004295493749329e-03 -9.999680482191979358e-01 -3.380376081852865672e-03 2.115612062706179408e+00
0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00
```

**注意**：
- Waymo 外参是 **camera-to-world** 变换
- 脚本会自动求逆得到 **world-to-camera**（DA3需要的格式）

---

## 🚀 使用方法

### 1. 修改路径配置

编辑 `inference_waymo.py`，修改以下路径：

```python
# 数据路径（修改为你的云端路径）
DATA_ROOT = "/home/zyc/street_crafter/data/waymo/049"

# 模型路径（修改为你的模型路径）
MODEL_PATH = "/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"
```

### 2. 配置推理参数

```python
# 推理配置
CAMERA_IDS = [0, 1, 2, 3, 4]  # Waymo 5个相机
FRAME_RANGE = (0, 10)         # 推理帧 0-9，设为 None 推理全部
UNDISTORT_IMAGES = True       # 是否去畸变

# 点云配置
GLB_CONF_THRESH_PERCENTILE = 5.0  # 置信度阈值（越低保留越多点）
GLB_NUM_MAX_POINTS = 5_000_000    # 最大点云数量
```

### 3. 运行推理

```bash
cd /home/zyc/Depth-Anything-3  # 或者你的工作目录
python inference_waymo.py
```

**注意**：
- `inference_waymo.py` 会**自动调用** `parse_waymo_calibration.py` 来解析标定文件
- **不需要**手动先运行 `parse_waymo_calibration.py`
- 脚本会在每次推理时自动加载和解析标定参数

---

## 📤 输出文件

每帧的输出保存在独立目录：

```
output/waymo_inference/
├── frame_000000/
│   ├── depths.npz                    # 深度图 + 置信度
│   ├── visualizations/               # 深度可视化
│   │   ├── frame_000000_camera_0.jpg
│   │   ├── frame_000000_camera_1.jpg
│   │   ├── frame_000000_camera_2.jpg
│   │   ├── frame_000000_camera_3.jpg
│   │   └── frame_000000_camera_4.jpg
│   └── glb/
│       └── scene.glb                 # 3D点云（可用Blender/MeshLab打开）
├── frame_000001/
│   └── ...
└── ...
```

### 深度图 NPZ 文件

```python
import numpy as np

data = np.load('output/waymo_inference/frame_000000/depths.npz')

depth = data['depth']          # (5, H, W) - 5个相机的深度图
conf = data['conf']            # (5, H, W) - 置信度
camera_names = data['camera_names']  # ['camera_0', ..., 'camera_4']
frame_id = data['frame_id']    # 0
```

---

## ⚙️ 重要参数说明

### 去畸变设置

```python
UNDISTORT_IMAGES = True  # 推荐开启，Waymo 相机有畸变
```

- **True**: 使用 `cv2.undistort()` 去畸变
- **False**: 直接使用原始图像

### 点云质量控制

```python
# 场景较小（<100m）
GLB_CONF_THRESH_PERCENTILE = 10.0
GLB_NUM_MAX_POINTS = 2_000_000

# 场景较大（>200m）- 推荐
GLB_CONF_THRESH_PERCENTILE = 5.0
GLB_NUM_MAX_POINTS = 5_000_000
```

### 批量推理范围

```python
# 推理前10帧
FRAME_RANGE = (0, 10)

# 推理第50-100帧
FRAME_RANGE = (50, 100)

# 推理所有帧
FRAME_RANGE = None
```

---

## 🔧 故障排除

### 1. 图像文件找不到

**错误**: `警告: 图像不存在，跳过`

**解决**: 检查图像命名格式是否为 `XXXXXX_Y.png`（6位帧号 + 相机ID）

### 2. 内参文件解析失败

**错误**: `IndexError: list index out of range`

**解决**: 确认内参文件有9行，格式正确

### 3. 外参矩阵形状错误

**错误**: `外参矩阵应为 4x4，实际为 (X, Y)`

**解决**: 确认外参文件是4行4列的矩阵

### 4. 相机位姿不正确

**问题**: GLB 中相机位置/朝向错误

**原因**: Waymo 外参的坐标系定义

**解决**:
- 检查外参是否为 camera-to-world
- 如果相机位姿反了，在 `parse_waymo_calibration.py` 中修改：
  ```python
  # 如果需要，改为不求逆
  # w2c = np.linalg.inv(ext)  # 注释掉
  w2c = ext  # 直接使用
  ```

### 5. 深度范围不合理

**问题**: 深度值过大或过小

**解决**:
1. 检查图像单位（米 vs 毫米）
2. 查看深度统计输出
3. 调整可视化归一化范围

---

## 📊 性能优化

### 批量处理建议

```python
# 小批量测试
FRAME_RANGE = (0, 5)

# 确认无误后批量处理
FRAME_RANGE = None
```

### 内存管理

- 每帧独立处理，避免内存积累
- 大场景可降低 `GLB_NUM_MAX_POINTS`

---

## 🎯 坐标系说明

### Waymo 坐标系

- **外参**: Camera → World (camera-to-world)
- **世界坐标系**: 通常是车辆坐标系或Lidar坐标系

### DA3 要求

- **外参**: World → Camera (world-to-camera)
- **脚本自动处理**: 求逆得到 w2c

### 验证方法

在 GLB 文件中：
- 相机应分布在车辆周围（surround view）
- 相机朝向应指向车辆外部
- 如果相反，需要调整求逆逻辑

---

## 📝 相关文件

- `parse_waymo_calibration.py` - 标定文件解析工具
- `inference_waymo.py` - 推理主脚本
- `WAYMO_INFERENCE_README.md` - 本文档

---

## 🔗 参考资料

- [Waymo Open Dataset](https://waymo.com/open/)
- [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
