# 车端周视相机 DA3 推理指南

针对车端环视相机系统（NoEER705_v3）的深度估计和3D重建。

---

## 📂 数据结构

### 服务器位置
```
/home/zyc/车端周视图/
├── NoEER705_v3/                # 标定文件
│   ├── camera/
│   │   ├── camera_01_intrinsics.yaml
│   │   ├── camera_01_extrinsics.yaml
│   │   ├── ... (camera_02 到 camera_11)
│   └── config.yaml
├── FN.jpg                      # 前窄角
├── FW.jpg                      # 前广角
├── FL.jpg                      # 前左
├── FR.jpg                      # 前右
├── RL.jpg                      # 左后
├── RR.jpg                      # 右后
└── RN.jpg                      # 后视
```

### 相机映射

| ID | 文件名 | 描述 | 类型 | 分辨率 |
|----|--------|------|------|--------|
| 1 | FN.jpg | 前窄角 (30度) | Pinhole | 3840x2160 |
| 2 | FW.jpg | 前广角 (120度) | Pinhole | 3840x2160 |
| 3 | FL.jpg | 前左 (120度) | Pinhole | 3840x2160 |
| 4 | FR.jpg | 前右 (120度) | Pinhole | 3840x2160 |
| 5 | RL.jpg | 左后 (60度) | Pinhole | 3840x2160 |
| 6 | RR.jpg | 右后 (60度) | Pinhole | 3840x2160 |
| 7 | RN.jpg | 后视 (60度) | Pinhole | 3840x2160 |

---

## 🚀 快速开始（三步走）

### 第1步：解析标定文件

```bash
cd ~/da3_wzh
python parse_vehicle_calibration.py
```

**功能**：
- 读取 NoEER705_v3 的 YAML 标定文件
- 将四元数转换为旋转矩阵
- 保存为 DA3 可用格式：`output/vehicle_surround_calibration/vehicle_camera_params.npz`

**输出**：
- `vehicle_camera_params.npz` - 相机参数（外参、内参、畸变）
- `vehicle_camera_params.txt` - 文本版本（便于查看）

### 第2步：确认模型路径

确保模型权重在正确位置：
```bash
ls /home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/
```

### 第3步：执行推理

```bash
python inference_vehicle_surround.py
```

**功能**：
- 加载7张环视图像
- 自动去畸变处理 ✅
- 生成深度图（NPZ）
- 生成3D点云（GLB）
- 生成可视化图像

**输出**：
- `da3_output/prediction.npz` - 深度数据
- `pointcloud/scene.glb` - 3D点云（车端环视）
- `visualizations/camera_*.png` - 各相机深度可视化
- `undistorted_images/*.jpg` - 去畸变后的图像

---

## 📋 标定文件格式

### YAML 格式说明

**内参文件** (`camera_XX_intrinsics.yaml`):
```yaml
type: pinhole
distortion_model: plumb_bob
width: 3840
height: 2160
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]  # 3x3内参矩阵（展平）
D: [k1, k2, p1, p2, k3]             # 5个畸变系数
```

**外参文件** (`camera_XX_extrinsics.yaml`):
```yaml
transform:
  rotation:          # 四元数 (x, y, z, w)
    x: 0.523836
    y: -0.503133
    z: 0.478676
    w: -0.493277
  translation:       # 平移向量 (米)
    x: 0.450986
    y: 0.165281
    z: -0.236292
label: camera_XX-lidar
```

**坐标系**：
- YAML 文件表示：**Camera → Lidar** 的变换（根据 label "camera_XX-lidar"）
- 解析脚本会自动求逆，得到 **Lidar → Camera** 的变换（DA3 需要的格式）
- 四元数格式：[x, y, z, w]（标量在最后）

---

## ⚙️ 配置选项

### 1. 去畸变设置

在 `inference_vehicle_surround.py` 中：

```python
UNDISTORT_IMAGES = True  # 车端图像需要去畸变
```

**说明**：车端相机通常有畸变，建议开启。

### 2. 处理分辨率

```python
PROCESS_RES = 504  # 可选: 392, 504, 616, 728
```

**建议**：504（平衡精度和速度）

### 3. 点云参数

```python
conf_thresh_percentile=10.0,  # 保留90%的点（车端场景）
num_max_points=3_000_000,     # 7个相机：300万点
```

**车端场景特点**：
- 深度范围小（0-50米）
- 近距离细节重要
- 可以保留更多点

---

## 📊 输出结果说明

### 1. NPZ 深度文件

**文件**: `output/vehicle_surround_inference/da3_output/prediction.npz`

**加载方式**:
```python
import numpy as np

data = np.load("output/vehicle_surround_inference/da3_output/prediction.npz")

depth = data['depth']          # (7, H, W) - 深度图，单位：米
extrinsics = data['extrinsics']  # (7, 4, 4) - 外参矩阵
intrinsics = data['intrinsics']  # (7, 3, 3) - 内参矩阵
conf = data['conf']            # (7, H, W) - 置信度
```

### 2. GLB 点云文件

**文件**: `output/vehicle_surround_inference/pointcloud/scene.glb`

**特点**：
- 360度车端环视点云
- 带彩色信息（来自图像）
- 显示7个相机位置和朝向
- 可用 3D 查看器打开

**查看方式**：
1. **在线查看**：https://gltf-viewer.donmccurdy.com/
2. **本地软件**：Blender, MeshLab, CloudCompare

### 3. 可视化图像

每个相机生成一张可视化图：
```
visualizations/
├── camera_01_camera_01.png  # 前窄角
├── camera_02_camera_02.png  # 前广角
├── ... (7个相机)
└── camera_07_camera_07.png  # 后视
```

**内容**：原始图像 + 深度图（并排显示）

---

## 🎯 车端场景特点

### 深度范围

| 区域 | 典型深度 | 说明 |
|------|---------|------|
| 车身周围 | 0-5米 | 近距离障碍物检测 |
| 车道内 | 5-30米 | 常规行驶场景 |
| 远处环境 | 30-50米 | 远距离感知 |

### 验证方法

检查深度值是否合理：
```bash
cat output/vehicle_surround_inference/visualizations/camera_01_camera_01.png
```

**合理范围**：
- 地面：1-2米
- 前方车辆：5-20米
- 建筑物：10-50米

---

## 🛠️ 故障排查

### 问题1: 找不到标定文件

**错误**:
```
错误: 找不到相机参数文件
```

**解决**:
1. 确认标定目录：`ls ./NoEER705_v3/camera/`
2. 运行解析脚本：`python parse_vehicle_calibration.py`

### 问题2: 图像找不到

**错误**:
```
警告: 找不到图像 /home/zyc/车端周视图/FN.jpg
```

**解决**:
修改 `inference_vehicle_surround.py` 中的 `DATA_DIR`：
```python
DATA_DIR = Path("/实际/路径/到/车端周视图")
```

### 问题3: 图像数量不匹配

**错误**:
```
警告: 图像数量(5)与相机数量(7)不匹配
```

**解决**:
1. 检查是否所有7张图像都存在
2. 检查文件名是否正确（FN.jpg, FW.jpg, ...）

### 问题4: 深度值异常

**症状**：深度值过大或过小

**解决**：
1. 检查标定文件是否正确
2. 确认坐标系定义（lidar-to-camera）
3. 验证畸变系数是否合理

---

## 📝 与路侧相机的区别

| 特性 | 车端周视 | 路侧固定 |
|------|---------|---------|
| 深度范围 | 0-50米 | 50-300米 |
| 畸变处理 | **必需** ✅ | 可选 |
| 相机数量 | 7个（环视） | 4个（定点） |
| 场景类型 | 近距离、360度 | 远距离、固定视角 |
| 置信度阈值 | 10%（宽松） | 5%（更宽松） |
| 最大点数 | 300万 | 500万 |

---

## 🔬 高级用法

### 1. 调整深度范围

针对特定场景调整可视化范围：

```python
# 在 visualize_surround_view 函数中
im_depth = ax2.imshow(depth, cmap='turbo', vmin=0, vmax=30)  # 0-30米
```

### 2. 只处理部分相机

```python
# 在 parse_vehicle_calibration.py 中
CAMERA_IDS = [1, 2, 7]  # 只处理前视和后视
```

### 3. 导出单独的点云

为每个相机生成独立点云：

```python
for i in range(len(images)):
    prediction_single = model.inference(
        image=[images[i]],
        extrinsics=extrinsics[i:i+1],
        intrinsics=intrinsics[i:i+1],
        export_dir=f"./output/camera_{i+1}_pointcloud",
        export_format="glb",
    )
```

### 4. 点云后处理

```python
import numpy as np

# 加载深度数据
data = np.load("output/vehicle_surround_inference/da3_output/prediction.npz")
depth = data['depth']  # (7, H, W)

# 过滤特定深度范围（例如：只保留0-20米）
mask = (depth > 0) & (depth < 20)
filtered_depth = np.where(mask, depth, 0)

# 保存过滤后的深度
np.savez("filtered_depth.npz", depth=filtered_depth)
```

---

## 📚 相关文档

- `parse_vehicle_calibration.py` - 标定文件解析脚本
- `inference_vehicle_surround.py` - 车端推理脚本
- `DA3_METRIC_DEPTH_EXPLAINED.md` - DA3 度量深度原理
- `CAMERA_POSE_INFERENCE_GUIDE.md` - 通用相机位姿指南

---

## 💡 提示

1. **首次运行**：先用1-2个相机测试，确认流程正确
2. **畸变处理**：车端相机畸变明显，务必开启去畸变
3. **深度范围**：车端场景深度小，关注近距离细节
4. **点云质量**：环视点云可能有接缝，这是正常现象

---

## ✉️ 问题反馈

如果遇到问题，请检查：
1. 标定文件格式是否正确（YAML）
2. 图像文件名是否匹配
3. 7张图像是否都存在
4. 畸变系数是否合理

如需帮助，请提供：
- 错误信息
- 使用的命令
- 图像和标定文件的路径
