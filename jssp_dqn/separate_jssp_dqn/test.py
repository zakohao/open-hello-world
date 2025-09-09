import os
import sys
import shutil

def get_size(path):
    """计算文件夹大小（字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError:
                pass
    return total_size

def format_size(size):
    """转为可读的 GB/MB"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"

# 常见路径
paths = {
    "CUDA Toolkit": r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "pip 缓存": os.path.expandvars(r"%LOCALAPPDATA%\pip\Cache"),
    "PyTorch (site-packages)": os.path.join(sys.prefix, "Lib", "site-packages", "torch"),
}

print("检查以下目录大小...\n")
for name, path in paths.items():
    if os.path.exists(path):
        size = get_size(path)
        print(f"{name}: {format_size(size)}  ({path})")
    else:
        print(f"{name}: 未找到 ({path})")

print("\n提示：")
print(" - CUDA Toolkit 通常 5-8GB/版本")
print(" - pip 缓存可安全清理: pip cache purge")
print(" - PyTorch 带 CUDA 的版本 2-4GB 很正常")
