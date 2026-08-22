"""
对图片按fx和fy顺序排序的脚本
将图片按照fx从-16到16，fy从0到16，phase从小到大的顺序排序
通过添加排序前缀实现，保持原文件名不变
"""
import os
import re

# 文件夹路径
folder_path = r"D:\桌面\毕设\fxfy[-16，16]"

# 正则表达式用于提取文件名中的fx、fy和phase值
pattern = r"fx=(-?\d+)_fy=(-?\d+)_phase=(\d+)"

# 获取文件夹中的所有图片文件
image_files = []
for filename in os.listdir(folder_path):
    if filename.endswith('.bmp'):
        # 提取fx、fy和phase值
        match = re.match(pattern, filename)
        if match:
            fx = int(match.group(1))
            fy = int(match.group(2))
            phase = int(match.group(3))
            image_files.append((fx, fy, phase, filename))

# 对文件进行排序：先按fx，再按fy，最后按phase
image_files.sort(key=lambda x: (x[0], x[1], x[2]))

print(f"找到 {len(image_files)} 个图片文件")
print("排序后的前10个文件：")
for i, (fx, fy, phase, filename) in enumerate(image_files[:10]):
    print(f"{i+1}. fx={fx}, fy={fy}, phase={phase}: {filename}")

# 生成排序前缀并重命名文件
print("\n开始排序...")
sorted_files = []
for i, (fx, fy, phase, filename) in enumerate(image_files):
    # 计算排序键：将fx从[-16,16]映射到[0,32]，fy从[0,16]映射到[0,16]
    fx_key = fx + 16
    fy_key = fy
    # 生成12位排序前缀：前4位是fx的键，中间4位是fy的键，最后4位是phase
    prefix = f"{fx_key:04d}{fy_key:04d}{phase:04d}"
    new_filename = f"{prefix}_{filename}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_filename)
    os.rename(old_path, new_path)
    sorted_files.append((new_filename, filename))

print(f"排序完成，共处理 {len(sorted_files)} 个文件")

# 显示排序结果
print("\n排序后的文件列表（前10个）：")
sorted_list = sorted(os.listdir(folder_path))
for filename in sorted_list[:10]:
    if filename.endswith('.bmp'):
        print(filename)

print("\n排序完成！文件现在按照fx从-16到16，fy从0到16的顺序显示。")
print("注意：文件名前添加了排序前缀，如需恢复原文件名，请运行恢复脚本。")
