#!/usr/bin/env python3
"""检查 DA3 目录结构并生成正确的导入路径"""

import sys
from pathlib import Path

script_path = Path(__file__).absolute()
script_dir = script_path.parent

print("=" * 70)
print("DA3 目录结构检查")
print("=" * 70)
print(f"脚本位置: {script_path}")
print(f"脚本目录: {script_dir}")
print()

# 检查可能的 DA3 源码位置
possible_locations = [
    script_dir / "depth_anything_3",
    script_dir / "src" / "depth_anything_3",
    script_dir.parent / "depth_anything_3",
    script_dir.parent / "src" / "depth_anything_3",
]

print("检查可能的 depth_anything_3 位置:")
print("-" * 70)
found = None
for loc in possible_locations:
    exists = loc.exists()
    dpt_exists = (loc / "dpt.py").exists() if exists else False
    marker = "✓" if dpt_exists else "✗"
    print(f"{marker} {loc}")
    if dpt_exists and found is None:
        found = loc.parent

print()
if found:
    print(f"✓ 找到 DA3 源码: {found}")
    print(f"\n建议的导入路径:")
    print(f'  sys.path.insert(0, "{found}")')
else:
    print("✗ 未找到 depth_anything_3 模块")
    print("\n请检查:")
    print("  1. 是否已克隆 Depth-Anything-3 仓库")
    print("  2. depth_anything_3/ 目录是否存在")
    print("  3. depth_anything_3/dpt.py 文件是否存在")

print()
print("当前 sys.path:")
print("-" * 70)
for p in sys.path[:5]:
    print(f"  {p}")
print()
