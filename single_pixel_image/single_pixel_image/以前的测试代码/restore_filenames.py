"""
恢复原始文件名的脚本
移除排序前缀，恢复到原始文件名
"""
import os
import re

# 文件夹路径
folder_path = r"D:\桌面\毕设\fxfy[-16，16]"

# 正则表达式用于提取原始文件名（去除排序前缀）
pattern = r"^\d+_(.*)"

# 获取文件夹中的所有图片文件
image_files = []
for filename in os.listdir(folder_path):
    if filename.endswith('.bmp'):
        # 检查是否有排序前缀
        match = re.match(pattern, filename)
        if match:
            original_filename = match.group(1)
            image_files.append((filename, original_filename))

print(f"找到 {len(image_files)} 个带排序前缀的文件")
print("恢复前的前10个文件：")
for i, (current_name, original_name) in enumerate(image_files[:10]):
    print(f"{i+1}. 当前: {current_name} → 原始: {original_name}")

# 恢复原始文件名
print("\n开始恢复...")
for current_name, original_name in image_files:
    old_path = os.path.join(folder_path, current_name)
    new_path = os.path.join(folder_path, original_name)
    os.rename(old_path, new_path)

print(f"恢复完成，共处理 {len(image_files)} 个文件")
print("\n恢复后的文件列表（前10个）：")
restored_list = os.listdir(folder_path)
restored_list.sort()
for filename in restored_list[:10]:
    if filename.endswith('.bmp'):
        print(filename)

print("\n恢复完成！文件已恢复到原始文件名。")
