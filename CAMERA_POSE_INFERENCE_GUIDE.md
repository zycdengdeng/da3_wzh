# DA3 相机位姿推理指南

## 目录
1. [模型权重设置](#1-模型权重设置)
2. [相机参数格式说明](#2-相机参数格式说明)
3. [快速开始](#3-快速开始)
4. [高级用法](#4-高级用法)
5. [常见问题](#5-常见问题)

---

## 1. 模型权重设置

### 问题：权重文件放错位置

您下载的权重在这里：
```
/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1
```

### 解决方案 1：移动到正确位置（推荐）

```bash
# 创建正确的目录结构
mkdir -p ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE

# 移动文件
mv /home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1/* \
   ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/

# 验证文件
ls ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/
# 应该看到: config.json, model.safetensors 等文件
```

### 解决方案 2：直接使用本地路径

在推理脚本中使用本地路径：

```python
MODEL_PATH = "/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1"
model = DepthAnything3.from_pretrained(MODEL_PATH)
```

### 解决方案 3：让代码自动下载（最简单）

```python
# 直接使用 Hugging Face 模型名称，会自动下载到正确位置
MODEL_PATH = "depth-anything/DA3NESTED-GIANT-LARGE"
model = DepthAnything3.from_pretrained(MODEL_PATH)
```

---

## 2. 相机参数格式说明

### 外参矩阵 (Extrinsics)

**格式**: `(N, 4, 4)` numpy 数组
**含义**: 世界坐标系到相机坐标系的变换矩阵

```
extrinsics[i] = [
    [r11, r12, r13, tx],    # 旋转矩阵 R (3x3) + 平移向量 t (3x1)
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0,   0,   0,   1 ]     # 齐次坐标
]
```

**示例代码**:
```python
import numpy as np

# 创建单位外参矩阵（相机在世界原点）
extrinsics = np.eye(4)

# 平移相机位置（向右移动 2 米）
extrinsics[0, 3] = 2.0

# 旋转相机（绕 Y 轴旋转 45 度）
import math
angle = math.radians(45)
extrinsics[0, 0] = math.cos(angle)
extrinsics[0, 2] = math.sin(angle)
extrinsics[2, 0] = -math.sin(angle)
extrinsics[2, 2] = math.cos(angle)
```

### 内参矩阵 (Intrinsics)

**格式**: `(N, 3, 3)` numpy 数组
**含义**: 相机内部参数（焦距、主点）

```
intrinsics[i] = [
    [fx,  0, cx],    # fx: X方向焦距, cx: 主点X坐标
    [ 0, fy, cy],    # fy: Y方向焦距, cy: 主点Y坐标
    [ 0,  0,  1]     # 齐次坐标
]
```

**参数说明**:
- `fx, fy`: 焦距（像素单位），通常 `fx ≈ fy`
- `cx, cy`: 主点坐标，通常在图像中心

**示例代码**:
```python
import numpy as np

image_width = 1920
image_height = 1080

# 创建内参矩阵
fx = fy = 1000.0  # 焦距（像素）
cx = image_width / 2.0
cy = image_height / 2.0

intrinsics = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])
```

**从视场角 (FOV) 计算焦距**:
```python
import math

# 已知水平视场角
fov_horizontal = 90  # 度
fx = image_width / (2 * math.tan(math.radians(fov_horizontal / 2)))

# 已知垂直视场角
fov_vertical = 60  # 度
fy = image_height / (2 * math.tan(math.radians(fov_vertical / 2)))
```

### 相机参数文件格式

**保存为 .npz 文件**:
```python
import numpy as np

# 多张图像的相机参数
extrinsics = np.stack([ext1, ext2, ext3])  # (3, 4, 4)
intrinsics = np.stack([int1, int2, int3])  # (3, 3, 3)

# 保存
np.savez(
    "camera_params.npz",
    extrinsics=extrinsics,
    intrinsics=intrinsics
)

# 加载
data = np.load("camera_params.npz")
extrinsics = data['extrinsics']
intrinsics = data['intrinsics']
```

---

## 3. 快速开始

### 基础用法：不提供相机参数

```python
from depth_anything_3.api import DepthAnything3
import torch

# 1. 加载模型
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)

# 2. 推理（模型会自动估计相机位姿）
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
prediction = model.inference(
    image=images,
    export_dir="./output"
)

# 3. 查看结果
print(f"深度图: {prediction.depth.shape}")
print(f"估计的外参: {prediction.extrinsics.shape}")
print(f"估计的内参: {prediction.intrinsics.shape}")
```

### 使用已知相机参数

```python
import numpy as np
from depth_anything_3.api import DepthAnything3

# 1. 准备相机参数
extrinsics = np.load("extrinsics.npy")  # (N, 4, 4)
intrinsics = np.load("intrinsics.npy")  # (N, 3, 3)

# 2. 加载模型
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device="cuda")

# 3. 使用相机参数进行推理
images = ["image1.jpg", "image2.jpg"]
prediction = model.inference(
    image=images,
    extrinsics=extrinsics,
    intrinsics=intrinsics,
    export_dir="./output"
)
```

### 使用提供的示例脚本

```bash
# 运行示例脚本
python inference_with_camera_pose.py
```

---

## 4. 高级用法

### 4.1 推理选项

```python
prediction = model.inference(
    image=images,
    extrinsics=extrinsics,
    intrinsics=intrinsics,

    # 处理参数
    process_res=504,                      # 处理分辨率（越大越精确，越慢）
    process_res_method="upper_bound_resize",  # 调整大小方法

    # 推理选项
    infer_metric=True,                    # 推理度量深度（真实尺度）
    infer_sky=True,                       # 推理天空分割
    infer_gs=False,                       # 推理高斯点云（需要 da3-giant）

    # 输出选项
    export_dir="./output",
    export_format="mini_npz",             # mini_npz, npz, glb, gs_ply, gs_video
    export_depth_vis=True,                # 导出深度可视化
    export_conf_vis=True,                 # 导出置信度可视化
    export_sky_vis=True,                  # 导出天空分割可视化

    # 对齐选项
    align_to_input_ext_scale=True,        # 将深度对齐到输入外参的尺度
)
```

### 4.2 输出格式说明

#### mini_npz（推荐，轻量级）
```python
# 保存内容：depth, conf, extrinsics, intrinsics
data = np.load("output/prediction.npz")
depth = data['depth']          # (N, H, W)
conf = data['conf']            # (N, H, W)
extrinsics = data['extrinsics']  # (N, 4, 4)
intrinsics = data['intrinsics']  # (N, 3, 3)
```

#### npz（完整）
```python
# 保存内容：depth, conf, extrinsics, intrinsics, images
data = np.load("output/prediction.npz")
images = data['images']  # (N, H, W, 3) 额外包含处理后的图像
```

#### glb（3D点云）
```
output/prediction.glb  # 可在 3D 查看器中打开
```

#### gs_ply（高斯点云）
```python
# 需要 infer_gs=True 和 da3-giant 模型
prediction = model.inference(
    image=images,
    infer_gs=True,
    export_format="gs_ply"
)
# 输出: output/prediction.ply
```

### 4.3 批量处理

```python
from pathlib import Path
import glob

# 批量处理目录中的所有图像
image_dir = Path("./路侧四个图")
image_paths = sorted(glob.glob(str(image_dir / "*.jpg")))

# 分批处理（每次处理 4 张图）
batch_size = 4
for i in range(0, len(image_paths), batch_size):
    batch_images = image_paths[i:i+batch_size]

    prediction = model.inference(
        image=batch_images,
        export_dir=f"./output/batch_{i//batch_size}"
    )

    print(f"批次 {i//batch_size} 完成")
```

### 4.4 路侧相机配置

```python
# 路侧相机通常有固定的安装位置和角度
def create_roadside_camera_params(num_cameras=4, height=5.0, spacing=10.0):
    """
    创建路侧相机参数

    Args:
        num_cameras: 相机数量
        height: 相机安装高度（米）
        spacing: 相机间距（米）
    """
    extrinsics = []
    intrinsics = []

    for i in range(num_cameras):
        # 外参：相机沿道路分布
        ext = np.eye(4)

        # 位置：沿道路方向间隔分布，固定高度
        ext[0, 3] = i * spacing  # X: 沿道路
        ext[1, 3] = height       # Y: 高度
        ext[2, 3] = 0.0          # Z: 道路中心线

        # 旋转：俯视角度（向下看 30 度）
        pitch = np.radians(30)  # 俯仰角
        ext[1, 1] = np.cos(pitch)
        ext[1, 2] = -np.sin(pitch)
        ext[2, 1] = np.sin(pitch)
        ext[2, 2] = np.cos(pitch)

        extrinsics.append(ext)

        # 内参：典型的路侧监控相机
        fx = fy = 1500.0  # 较大焦距（长焦）
        cx = 1920 / 2.0
        cy = 1080 / 2.0

        intr = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        intrinsics.append(intr)

    return np.stack(extrinsics), np.stack(intrinsics)

# 使用
extrinsics, intrinsics = create_roadside_camera_params(num_cameras=4)
```

---

## 5. 常见问题

### Q1: 权重文件放错位置怎么办？

**A**: 有三种解决方法：
1. **移动文件到正确位置**（见 [模型权重设置](#1-模型权重设置)）
2. **直接使用本地路径**加载
3. **重新下载**到正确位置

```bash
# 推荐：移动到正确位置
mkdir -p ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE
mv /home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1/* \
   ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/
```

### Q2: 如何知道我的相机参数？

**A**: 获取相机参数的方法：

1. **相机标定**（最准确）：
   - 使用 OpenCV 的相机标定工具
   - 使用棋盘格标定板

2. **从相机规格计算**：
   ```python
   # 已知传感器尺寸和焦距
   sensor_width = 6.4  # mm
   focal_length = 3.6  # mm
   image_width = 1920  # pixels

   fx = (focal_length / sensor_width) * image_width
   ```

3. **从 EXIF 信息提取**：
   ```python
   from PIL import Image
   from PIL.ExifTags import TAGS

   img = Image.open("image.jpg")
   exif = img._getexif()
   focal_length = exif.get(37386)  # FocalLength
   ```

4. **让模型自动估计**：
   ```python
   # 不提供相机参数，模型会自动估计
   prediction = model.inference(image=images)
   estimated_intrinsics = prediction.intrinsics
   ```

### Q3: 深度值是什么单位？

**A**: 取决于是否使用度量深度：

- **相对深度** (`infer_metric=False`): 无单位，相对深度值
- **度量深度** (`infer_metric=True`): **米** (meters)

```python
# 度量深度（真实尺度）
prediction = model.inference(
    image=images,
    infer_metric=True  # 深度单位为米
)

# 获取深度统计
print(f"最小深度: {prediction.depth.min():.2f} 米")
print(f"最大深度: {prediction.depth.max():.2f} 米")
print(f"平均深度: {prediction.depth.mean():.2f} 米")
```

### Q4: 如何提高推理速度？

**A**: 几种加速方法：

1. **降低处理分辨率**：
   ```python
   prediction = model.inference(
       image=images,
       process_res=392  # 默认 504，降低到 392
   )
   ```

2. **使用更小的模型**：
   ```python
   # da3-large (0.35B) 而不是 da3-giant (1.15B)
   model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
   ```

3. **禁用不需要的推理**：
   ```python
   prediction = model.inference(
       image=images,
       infer_sky=False,   # 不需要天空分割
       infer_gs=False,    # 不需要高斯点云
   )
   ```

4. **使用半精度**：
   ```python
   model = model.half()  # FP16
   ```

### Q5: 外参矩阵是 world-to-camera 还是 camera-to-world？

**A**: **world-to-camera**

```python
# 正确的外参格式
world_point = np.array([x, y, z, 1])
camera_point = extrinsics @ world_point

# 如果你有 camera-to-world (c2w)，需要求逆
c2w = np.array([...])  # camera-to-world
w2c = np.linalg.inv(c2w)  # world-to-camera
extrinsics = w2c
```

### Q6: 如何可视化深度图？

**A**: 使用内置的可视化功能：

```python
# 方法1：自动导出可视化
prediction = model.inference(
    image=images,
    export_dir="./output",
    export_depth_vis=True,  # 导出深度可视化
    export_conf_vis=True,   # 导出置信度可视化
)
# 输出: output/depth_vis_000.png, output/depth_vis_001.png, ...

# 方法2：手动可视化
import matplotlib.pyplot as plt

depth = prediction.depth[0]  # 第一张图的深度
plt.imshow(depth, cmap='turbo')
plt.colorbar()
plt.savefig("depth_vis.png")
```

---

## 参考资料

- [DA3 官方文档](docs/API.md)
- [DA3 CLI 文档](docs/CLI.md)
- [相机标定教程](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

---

## 联系与支持

如有问题，请查看：
- GitHub Issues: https://github.com/depth-anything/DA3
- Hugging Face: https://huggingface.co/depth-anything
