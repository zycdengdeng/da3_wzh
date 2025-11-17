# 真实路侧相机数据推理指南

本指南针对您的真实路侧相机数据（4个相机 + 标定文件）进行 DA3 深度估计。

---

## 📂 您的数据结构

```
/home/zyc/路侧四个图/
├── calib.json                    # 标定文件
├── cam0_1742877424148.png        # 相机0图像
├── cam3_1742877424148.png        # 相机3图像
├── cam6_1742877424148.png        # 相机6图像
└── cam9_1742877424148.png        # 相机9图像
```

**标定文件包含**：
- 4个非鱼眼相机 (cam0, cam3, cam6, cam9)
- 每个相机的内参矩阵 (3x3)
- 每个相机的畸变参数 (5个系数)
- 每个相机相对于虚拟雷达坐标系的外参（旋转+平移）

---

## 🚀 快速开始（三步走）

### 第1步：解析标定文件

```bash
python parse_calibration.py
```

**功能**：
- 读取 `calib.json` 文件
- 提取相机 0, 3, 6, 9 的参数
- 将罗德里格斯旋转向量转换为旋转矩阵
- 保存为 DA3 可用的格式：`output/calibration/camera_params.npz`

**输出**：
- `output/calibration/camera_params.npz` - 相机参数（外参、内参、畸变）
- `output/calibration/camera_params.txt` - 参数文本版本（便于查看）
- 终端显示相机空间布局和间距信息

### 第2步：修正模型权重路径

您下载的权重在错误的位置，需要修正：

**方案A：移动到正确位置（推荐）**
```bash
mkdir -p ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE
mv /home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1/* \
   ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/
```

**方案B：修改脚本使用本地路径**

编辑 `inference_with_real_calibration.py`：
```python
# 第 252 行左右
MODEL_PATH = "/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1"
```

### 第3步：执行推理

```bash
python inference_with_real_calibration.py
```

**功能**：
- 加载4张图像和标定参数
- 对图像进行去畸变处理（可选）
- 使用 DA3 估计深度
- 生成可视化和统计报告

**输出**：
- `output/real_calibration_inference/da3_output/prediction.npz` - 深度预测结果
- `output/real_calibration_inference/visualizations/result_*.png` - 可视化图像
- `output/real_calibration_inference/depth_statistics_report.txt` - 统计报告
- `output/real_calibration_inference/undistorted_images/` - 去畸变后的图像

---

## 📝 详细步骤说明

### 标定文件格式

您的标定文件使用以下格式：

```json
{
  "camera": {
    "0": {
      "intri": [fx, 0, cx, 0, fy, cy, 0, 0, 1],  // 内参矩阵（展平）
      "distor": [k1, k2, p1, p2, k3],            // 畸变系数
      "virtualLidarToCam": {
        "rotate": [rx, ry, rz],                  // 罗德里格斯旋转向量
        "trans": [tx, ty, tz]                    // 平移向量（米）
      }
    }
  }
}
```

**参数说明**：
- `intri`: 相机内参矩阵（3x3），包含焦距 (fx, fy) 和主点 (cx, cy)
- `distor`: 径向和切向畸变系数 (k1, k2, p1, p2, k3)
- `rotate`: 罗德里格斯旋转向量（3D），需转换为旋转矩阵
- `trans`: 虚拟雷达到相机的平移（单位：米）

### 坐标系说明

```
虚拟雷达坐标系 (Virtual Lidar)
    ↓ (virtualLidarToCam)
相机坐标系 (Camera)
```

在推理时，我们将**虚拟雷达坐标系**视为**世界坐标系**，因此：
- `extrinsics[i]` = `virtualLidarToCam[i]` (4x4 齐次变换矩阵)

### 相机布局（从您的数据）

根据标定文件，4个相机的空间布局大致如下：

```
                     ┌─────────┐
                     │  Cam 6  │  (远距离，高位)
                     └─────────┘

    ┌─────────┐                    ┌─────────┐
    │  Cam 0  │                    │  Cam 9  │
    └─────────┘                    └─────────┘

                     ┌─────────┐
                     │  Cam 3  │  (近距离，低位)
                     └─────────┘
```

**实际位置**（相对于虚拟雷达）：
- Cam 0: (-69.27, -6.28, 34.07) 米
- Cam 3: (9.47, 4.21, 1.82) 米
- Cam 6: (49.58, -18.72, 74.60) 米
- Cam 9: (-18.86, -30.36, 125.04) 米

---

## ⚙️ 配置选项

### 图像去畸变

在 `inference_with_real_calibration.py` 中：

```python
# 第 258 行左右
UNDISTORT_IMAGES = True   # True: 去畸变，False: 使用原始图像
```

**建议**：
- 如果图像畸变明显（边缘弯曲） → 设为 `True`
- 如果图像畸变很小 → 可设为 `False`（更快）

### 处理分辨率

```python
# 第 259 行左右
PROCESS_RES = 504  # 可选值: 392, 504, 616, 728
```

**说明**：
- 更高分辨率 → 更精确，但更慢
- 更低分辨率 → 更快，但稍微不精确
- 推荐值：504（平衡精度和速度）

### 推理选项

在 `inference_with_real_calibration.py` 的 `model.inference()` 调用中：

```python
prediction = model.inference(
    image=images_for_inference,
    extrinsics=extrinsics,      # 使用标定的外参
    intrinsics=intrinsics,      # 使用标定的内参
    process_res=PROCESS_RES,
    infer_metric=True,          # True: 度量深度（米），False: 相对深度
    infer_sky=True,             # True: 天空分割，False: 仅深度
    export_format="mini_npz",   # npz, mini_npz, glb, gs_ply
    export_depth_vis=True,      # 导出深度可视化
    export_sky_vis=True,        # 导出天空可视化
    export_conf_vis=True,       # 导出置信度可视化
)
```

---

## 📊 输出结果说明

### 1. NPZ 文件（主要输出）

**文件**: `output/real_calibration_inference/da3_output/prediction.npz`

**加载方式**:
```python
import numpy as np

data = np.load("output/real_calibration_inference/da3_output/prediction.npz")

depth = data['depth']          # (4, H, W) - 深度图，单位：米
conf = data['conf']            # (4, H, W) - 置信度 [0-1]
extrinsics = data['extrinsics']  # (4, 4, 4) - 外参矩阵
intrinsics = data['intrinsics']  # (4, 3, 3) - 内参矩阵

# 如果有天空分割
sky = data['sky']              # (4, H, W) - 天空掩码 [0-1]
```

**深度值含义**：
- 单位：**米 (meters)**
- 表示从相机到场景点的距离
- 例如：`depth[0, 100, 200] = 25.5` 表示该像素点距离相机 0 为 25.5 米

### 2. 可视化图像

**目录**: `output/real_calibration_inference/visualizations/`

每个相机生成一个可视化文件：
- `result_00_cam0_*.png` - 相机0的4宫格可视化
- `result_01_cam3_*.png` - 相机3的4宫格可视化
- `result_02_cam6_*.png` - 相机6的4宫格可视化
- `result_03_cam9_*.png` - 相机9的4宫格可视化

**4宫格内容**：
```
┌──────────────┬──────────────┐
│ 原始图像      │  深度图       │
├──────────────┼──────────────┤
│ 天空分割      │  置信度图     │
└──────────────┴──────────────┘
```

### 3. 统计报告

**文件**: `output/real_calibration_inference/depth_statistics_report.txt`

**内容**：
- 每个相机的深度统计（最小值、最大值、平均值、中位数、标准差）
- 地面区域的深度统计（排除天空）
- 天空占比
- 深度百分位数 (10%, 25%, 50%, 75%, 90%)

---

## 🔍 结果验证

### 检查深度范围是否合理

路侧监控场景的典型深度范围：
- **最小深度**: 5-20 米（近处的道路或车辆）
- **最大深度**: 50-150 米（远处的场景）
- **平均深度**: 30-70 米

如果您的结果远超这个范围，可能需要调整参数。

### 检查天空分割

查看可视化图像，确认：
- 天空区域被正确识别（白色）
- 地面/建筑被正确识别（黑色）

### 检查置信度

高置信度（亮色）应该出现在：
- 纹理丰富的区域（建筑、车辆）
- 清晰的边缘

低置信度（暗色）可能出现在：
- 均匀区域（天空、墙面）
- 重复纹理（窗户）

---

## 🛠️ 故障排查

### 问题1: 找不到标定文件

**错误**:
```
错误: 找不到标定文件 /home/zyc/路侧四个图/calib.json
```

**解决**:
编辑 `parse_calibration.py` 或 `inference_with_real_calibration.py`，修改路径：
```python
DATA_DIR = Path("/实际/路径/到/路侧四个图")
```

### 问题2: 找不到模型权重

**错误**:
```
✗ 模型加载失败
```

**解决**:
参考 [第2步：修正模型权重路径](#第2步修正模型权重路径)

### 问题3: CUDA out of memory

**错误**:
```
RuntimeError: CUDA out of memory
```

**解决**:
1. 降低处理分辨率：
   ```python
   PROCESS_RES = 392  # 从 504 降到 392
   ```

2. 或使用 CPU（慢）：
   ```python
   device = torch.device("cpu")
   ```

### 问题4: 深度值看起来不合理

**可能原因**:
1. 相机参数错误
2. 坐标系定义不一致
3. 需要尺度校正

**调试方法**:
1. 检查 `output/calibration/camera_params.txt` 确认参数正确
2. 尝试不使用相机参数，让模型自动估计：
   ```python
   prediction = model.inference(
       image=images,
       extrinsics=None,  # 不提供
       intrinsics=None,  # 不提供
       ...
   )
   ```

---

## 📚 进阶用法

### 1. 只处理部分相机

修改 `parse_calibration.py`:
```python
CAMERA_IDS = ["0", "6"]  # 只处理相机 0 和 6
```

修改 `inference_with_real_calibration.py`:
```python
CAMERA_IDS = ["0", "6"]
```

### 2. 批量处理多个时间戳的数据

如果您有多组图像（不同时间戳），创建循环：

```python
timestamps = ["1742877424148", "1742877424200", "1742877424252"]

for ts in timestamps:
    IMAGE_PATTERN = f"cam{{}}_{ts}.png"
    # ... 执行推理
```

### 3. 导出3D点云

修改推理调用：
```python
prediction = model.inference(
    ...,
    export_format="glb",  # 导出 .glb 格式
)
```

然后使用 3D 查看器（如 Blender, MeshLab）打开 `output/*/prediction.glb`

### 4. 使用深度数据进行后处理

```python
import numpy as np

# 加载深度数据
data = np.load("output/real_calibration_inference/da3_output/prediction.npz")
depth = data['depth']  # (4, H, W)
intrinsics = data['intrinsics']  # (4, 3, 3)

# 示例：将深度图转换为3D点云
def depth_to_pointcloud(depth, intrinsics):
    """将深度图转换为3D点云"""
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 组合成点云 (N, 3)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    return points

# 处理每个相机
for i in range(4):
    points = depth_to_pointcloud(depth[i], intrinsics[i])
    np.save(f"pointcloud_cam{i}.npy", points)
```

---

## 📖 相关文档

- `CAMERA_POSE_INFERENCE_GUIDE.md` - 通用相机位姿推理指南
- `inference_with_camera_pose.py` - 通用推理脚本示例
- `roadside_camera_example.py` - 路侧相机示例脚本
- `docs/API.md` - DA3 API 完整文档

---

## 💡 提示

1. **首次运行**：建议先用一张图像测试，确认流程正确后再处理全部
2. **保存中间结果**：去畸变后的图像会被保存，下次可以直接使用
3. **参数调优**：如果结果不理想，尝试调整 `process_res` 或 `infer_metric`
4. **性能优化**：如果处理速度慢，考虑使用更小的模型（如 da3-large 而不是 da3-nested-giant-large）

---

## ✉️ 问题反馈

如果遇到问题，请检查：
1. 所有路径是否正确
2. 模型权重是否正确加载
3. 图像和标定文件是否匹配
4. GPU 内存是否充足

如需帮助，请提供：
- 错误信息的完整输出
- 使用的命令
- 数据文件的路径和大小
