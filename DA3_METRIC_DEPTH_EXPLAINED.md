# DA3 度量深度估计原理

## 🎯 核心问题

**DA3 如何估计绝对深度（度量深度）？是靠相机位姿还是模型猜测？**

---

## 📊 答案：主要靠独立的度量分支 + 最小二乘对齐

DA3 使用 **Nested 双分支架构** 来实现度量深度估计：

### 1. 双分支架构

```
输入图像
    ├── 分支1: 任意视角分支 (da3)
    │   ├── 输入: 图像 + 相机位姿 (extrinsics/intrinsics)
    │   └── 输出: 相对深度 (多视角一致)
    │
    └── 分支2: 度量分支 (da3_metric)
        ├── 输入: 仅图像 (不使用位姿!)
        └── 输出: 度量深度 (真实尺度)
```

### 2. 关键代码分析

**文件**: `src/depth_anything_3/model/da3.py` (第276-352行)

#### 步骤1: 两个分支独立推理

```python
def forward(self, x, extrinsics, intrinsics, ...):
    # 分支1：使用位姿的任意视角深度
    output = self.da3(x, extrinsics, intrinsics, ...)

    # 分支2：不使用位姿的度量深度 (关键！)
    metric_output = self.da3_metric(x, infer_gs=infer_gs)  # 注意没有传位姿
```

**分析**：
- 度量分支 `da3_metric` **完全不使用相机位姿**
- 它通过深度学习**直接从图像猜测真实尺度**
- 这是通过在大规模度量深度数据集上训练实现的

#### 步骤2: 度量缩放（使用内参）

```python
def _apply_metric_scaling(self, output, metric_output):
    # 使用相机内参对度量深度进行缩放
    metric_output.depth = apply_metric_scaling(
        metric_output.depth,
        output.intrinsics,  # 使用内参调整焦距相关的尺度
    )
    return output
```

**分析**：
- 内参（焦距）用于调整度量深度的尺度
- 这是因为不同焦距会影响深度的感知

#### 步骤3: 最小二乘对齐

```python
def _apply_depth_alignment(self, output, metric_output):
    # 计算非天空区域的掩码
    non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

    # 在高置信度区域计算最优尺度因子
    align_mask = compute_alignment_mask(
        output.depth_conf, non_sky_mask, output.depth, metric_output.depth, ...
    )

    # 最小二乘求解尺度因子
    valid_depth = output.depth[align_mask]
    valid_metric_depth = metric_output.depth[align_mask]
    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

    # 同时缩放深度和外参的平移部分
    output.depth *= scale_factor
    output.extrinsics[:, :, :3, 3] *= scale_factor  # 平移向量也要缩放
    output.is_metric = 1

    return output
```

**分析**：
- 将任意视角分支的相对深度对齐到度量分支的绝对尺度
- 通过最小二乘法求解最优缩放因子
- **同时调整外参的平移向量**，保证位姿和深度尺度一致

---

## 🔬 总结

### DA3 度量深度估计依赖于：

| 组件 | 作用 | 是否必需 |
|------|------|---------|
| **度量分支** | 直接猜测真实深度尺度 | ✅ **核心** |
| **相机内参** | 焦距相关的尺度调整 | ✅ 必需 |
| **相机外参** | 多视角一致性（任意视角分支） | ⚠️ 可选（提高质量） |
| **训练数据** | 度量深度监督信号 | ✅ **核心** |

### 结论

DA3 的度量深度是：
1. **主要靠模型猜测**：度量分支通过深度学习从图像特征直接估计真实尺度
2. **部分靠内参**：使用焦距信息调整尺度
3. **不直接靠外参**：外参主要用于多视角一致性（任意视角分支），不直接用于度量估计

**核心能力来源**：
- 在大规模度量深度数据集上训练（如 Hypersim, Virtual KITTI）
- 学习到了"真实世界尺度"的先验知识
- 例如：识别到"这是一辆车" → 猜测其大约 4-5 米长

---

## 🎯 实际应用建议

### 1. 提高度量深度准确性

**有帮助**：
- ✅ 提供准确的相机内参（焦距）
- ✅ 场景中有常见物体（车、人、建筑）作为尺度参考
- ✅ 提供相机位姿（提高多视角一致性）

**帮助有限**：
- ⚠️ 仅提供外参（度量分支不直接使用）
- ⚠️ 极端特殊场景（模型未见过的尺度）

### 2. 验证度量深度

可以通过以下方式验证：
```python
# 加载结果
data = np.load("output/real_calibration_inference/da3_output/prediction.npz")
depth = data['depth']  # 单位：米

# 检查已知物体的深度是否合理
# 例如：路侧相机到对面建筑的距离应该是 30-50 米
# 如果模型估计为 100 米，说明尺度有偏差
```

### 3. 与 VGGT 的对比

您提到的 VGGT 可能使用不同的度量估计策略：
- **VGGT**: 可能通过 depth config 显式设置深度范围
- **DA3**: 通过度量分支隐式学习深度尺度

DA3 的优势是**自动**且**鲁棒**，但在极端场景下可能需要手动调整。

---

## 📚 相关代码位置

- **Nested架构**: `src/depth_anything_3/model/da3.py:248-352`
- **度量缩放**: `src/depth_anything_3/utils/alignment.py`
- **最小二乘对齐**: `least_squares_scale_scalar` 函数
- **GLB导出**: `src/depth_anything_3/utils/export/glb.py`

---

## 💡 关于 GLB 范围限制问题

您的 GLB 点云范围只有 40 米的问题，**不是度量估计的问题**，而是：

### 问题原因

1. **置信度过滤过严**：
   - 原设置：`conf_thresh_percentile=40.0` → 只保留前60%高置信度的点
   - 远距离点置信度低 → 被过滤掉

2. **点数限制**：
   - 原设置：`num_max_points=2_000_000` → 随机下采样可能丢失远处点

### 解决方案

已在 `inference_with_real_calibration.py` 中修改：

```python
conf_thresh_percentile=5.0,    # 40.0 → 5.0 (保留95%的点)
num_max_points=5_000_000,      # 2M → 5M (增加点数上限)
```

这样可以保留更多远距离的点，重建范围应该能达到 200-300 米。

---

## 🚀 使用建议

1. **重新生成 GLB**：
   ```bash
   cd ~/Depth-Anything-3
   git pull
   python inference_with_real_calibration.py
   ```

2. **查看深度统计**：
   - 检查 `depth_statistics_report.txt`
   - 确认深度范围是否符合预期（应该有 200-300 米的点）

3. **验证 GLB**：
   - 在 3D 查看器中打开 `prediction.glb`
   - 确认远距离建筑物是否被重建出来

4. **微调参数**（如果需要）：
   - 进一步降低 `conf_thresh_percentile`（如 2.0, 1.0）
   - 增加 `num_max_points`（如 10_000_000）
