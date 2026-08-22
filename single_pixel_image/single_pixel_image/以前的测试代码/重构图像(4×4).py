"""
使用傅里叶系数重构128×128的图片
输入文件：处理2.xlsx（包含傅里叶系数）
输出：重构的图片
"""
import pandas as pd
import numpy as np
from PIL import Image

# 读取处理后的Excel文件
input_file = r"D:\桌面\毕设\data[16,16]\数据13的处理.xlsx"
print(f"正在读取Excel文件: {input_file}")

df = pd.read_excel(input_file)

# 创建5x5的傅里叶系数矩阵（fx和fy从0到4）
fourier_matrix_5x5 = np.zeros((5, 5), dtype=complex)

# 填充傅里叶系数矩阵
for i, row in df.iterrows():
    fx_fy = row['fx/fy']
    # 解析fx和fy
    parts = fx_fy.split(', ')
    fx = int(parts[0])
    fy = int(parts[1])
    
    # 确保fx和fy在0-4范围内
    if 0 <= fx < 5 and 0 <= fy < 5:
        # 获取傅里叶系数
        fourier_str = row['傅里叶系数']
        
        if pd.notna(fourier_str) and fourier_str != '':
            # 解析复数
            if 'j' in fourier_str:
                # 有虚部的情况
                if '+' in fourier_str:
                    parts = fourier_str.split(' + ')
                    real_part = float(parts[0])
                    imag_part = float(parts[1].replace('j', ''))
                elif '-' in fourier_str:
                    parts = fourier_str.split(' - ')
                    real_part = float(parts[0])
                    imag_part = -float(parts[1].replace('j', ''))
                fourier_matrix_5x5[fx, fy] = complex(real_part, imag_part)
            else:
                # 只有实部的情况
                fourier_matrix_5x5[fx, fy] = complex(float(fourier_str), 0)

# 处理(0,0)位置的特殊情况
# 从D3列获取(0,0)的值（直流分量）
if pd.notna(df.iloc[0]['D3 (180°)']) and not isinstance(df.iloc[0]['D3 (180°)'], str):
    fourier_matrix_5x5[0, 0] = complex(float(df.iloc[0]['D3 (180°)']), 0)

print("5x5傅里叶系数矩阵：")
print(fourier_matrix_5x5)

# 创建128x128的傅里叶系数矩阵
image_size = 128
fourier_matrix_128x128 = np.zeros((image_size, image_size), dtype=complex)

# 填充128x128矩阵，只在fx和fy为0-4的位置设置值
for fx in range(5):
    for fy in range(5):
        fourier_matrix_128x128[fx, fy] = fourier_matrix_5x5[fx, fy]

# 进行逆傅里叶变换
print("执行逆傅里叶变换...")
reconstructed_image = np.fft.ifft2(fourier_matrix_128x128)

# 获取实部作为重构的图像
reconstructed_image_real = np.real(reconstructed_image)

# 归一化到0-255范围
print("归一化图像数据...")
min_val = np.min(reconstructed_image_real)
max_val = np.max(reconstructed_image_real)

if max_val > min_val:
    reconstructed_image_normalized = ((reconstructed_image_real - min_val) / (max_val - min_val)) * 255
else:
    reconstructed_image_normalized = np.zeros_like(reconstructed_image_real)

reconstructed_image_normalized = reconstructed_image_normalized.astype(np.uint8)

print("\n重构的128x128图像数据范围：")
print(f"最小值: {np.min(reconstructed_image_normalized)}")
print(f"最大值: {np.max(reconstructed_image_normalized)}")
print(f"平均值: {np.mean(reconstructed_image_normalized)}")

# 保存图像
output_image_path = r"D:\桌面\毕设\data[16,16]\处理数据1重构的数字1图像.png"
print(f"\n保存重构的图片到: {output_image_path}")

img = Image.fromarray(reconstructed_image_normalized, mode='L')
img.save(output_image_path)

print("重构完成！")
