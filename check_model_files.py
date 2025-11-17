#!/usr/bin/env python3
"""
检查模型文件位置并提供修复建议
"""

from pathlib import Path
import os

def check_model_files():
    """检查模型文件位置"""

    print("=" * 70)
    print("检查 DA3 模型文件位置")
    print("=" * 70)

    # 可能的位置列表
    possible_paths = [
        Path.home() / ".cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE",
        Path.home() / ".cache/huggingface/models/nvidia/Cosmos-Guardrail1",
        Path("/home/zyc/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE"),
        Path("/home/zyc/.cache/huggingface/models/nvidia/Cosmos-Guardrail1"),
    ]

    print("\n检查可能的模型位置:")
    print("-" * 70)

    found_paths = []

    for path in possible_paths:
        exists = path.exists()
        print(f"\n路径: {path}")
        print(f"  存在: {'✓' if exists else '✗'}")

        if exists:
            # 检查目录内容
            try:
                files = list(path.iterdir())
                print(f"  文件数量: {len(files)}")

                # 查找关键文件
                key_files = {
                    "config.json": any(f.name == "config.json" for f in files),
                    "model.safetensors": any(f.name == "model.safetensors" for f in files),
                    "model.pth": any(f.name == "model.pth" for f in files),
                }

                print(f"  关键文件:")
                for fname, found in key_files.items():
                    print(f"    - {fname}: {'✓' if found else '✗'}")

                if any(key_files.values()):
                    found_paths.append((path, key_files))

                # 列出所有文件
                print(f"  所有文件:")
                for f in sorted(files)[:10]:  # 只显示前10个
                    size = f.stat().st_size / (1024**3) if f.is_file() else 0
                    print(f"    - {f.name} ({size:.2f} GB)" if f.is_file() else f"    - {f.name}/ (dir)")

            except Exception as e:
                print(f"  错误: {e}")

    # 总结和建议
    print("\n" + "=" * 70)
    print("诊断结果")
    print("=" * 70)

    if not found_paths:
        print("\n✗ 未找到模型文件！")
        print("\n建议:")
        print("1. 检查您执行 'mv' 命令后的实际文件位置:")
        print("   ls -la ~/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE/")
        print()
        print("2. 如果文件在 Cosmos-Guardrail1 目录，使用以下命令:")
        print("   export MODEL_PATH=~/.cache/huggingface/models/nvidia/Cosmos-Guardrail1")
        print("   # 然后在脚本中使用 os.getenv('MODEL_PATH')")

    else:
        print(f"\n✓ 找到 {len(found_paths)} 个可能的模型位置:")
        for i, (path, key_files) in enumerate(found_paths, 1):
            print(f"\n  [{i}] {path}")
            has_config = key_files.get("config.json", False)
            has_model = key_files.get("model.safetensors", False) or key_files.get("model.pth", False)

            if has_config and has_model:
                print(f"      状态: ✓ 完整（可直接使用）")
                print(f"      建议在脚本中使用:")
                print(f'      MODEL_PATH = "{path}"')
            else:
                print(f"      状态: ⚠ 不完整")
                if not has_config:
                    print(f"      缺少: config.json")
                if not has_model:
                    print(f"      缺少: model.safetensors 或 model.pth")

    # 检查环境变量
    print("\n" + "-" * 70)
    print("环境变量检查")
    print("-" * 70)

    hf_home = os.getenv("HF_HOME")
    hf_hub_cache = os.getenv("HF_HUB_CACHE")

    print(f"HF_HOME: {hf_home if hf_home else '(未设置)'}")
    print(f"HF_HUB_CACHE: {hf_hub_cache if hf_hub_cache else '(未设置)'}")
    print(f"HOME: {Path.home()}")

    # 生成修复脚本
    print("\n" + "=" * 70)
    print("生成修复命令")
    print("=" * 70)

    if found_paths:
        best_path = found_paths[0][0]
        print(f"\n在 inference_with_real_calibration.py 中修改:")
        print(f'MODEL_PATH = "{best_path}"')
    else:
        print("\n如果您确定文件已下载，请手动查找:")
        print("find ~/ -name 'config.json' -path '*depth-anything*' 2>/dev/null")
        print("find ~/ -name 'model.safetensors' -size +1G 2>/dev/null")


if __name__ == "__main__":
    check_model_files()
