"""
用处理后的excel表格中的数据进行图像的傅里叶重构
读取0-16范围的傅里叶系数，重构128x128图像
"""
import pandas as pd
import numpy as np
from PIL import Image

# 读取处理后的数据
input_file = r"D:\桌面\毕设\data[16,16]\数据16的处理.xlsx"
df = pd.read_excel(input_file)

# 创建17x17的傅里叶系数矩阵（fx和fy从0到16）
fourier_matrix_17x17 = np.zeros((17, 17), dtype=complex)

# 填充傅里叶系数矩阵
for i, row in df.iterrows():
    fx_fy = row['fx/fy']
    # 解析fx和fy
    parts = fx_fy.split(', ')
    fx = int(parts[0])
    fy = int(parts[1])
    
    # 确保fx和fy在0-16范围内
    if 0 <= fx < 17 and 0 <= fy < 17:
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
                # 交换fx和fy的顺序，解决图像翻转问题
                fourier_matrix_17x17[fy, fx] = complex(real_part, imag_part)
            else:
                # 只有实部的情况
                # 交换fx和fy的顺序，解决图像翻转问题
                fourier_matrix_17x17[fy, fx] = complex(float(fourier_str), 0)

# 处理(0,0)位置的特殊情况
# 从D3列获取(0,0)的值（直流分量）
if pd.notna(df.iloc[0]['D3 (180°)']) and not isinstance(df.iloc[0]['D3 (180°)'], str):
    # (0,0)位置交换后仍然是(0,0)
    fourier_matrix_17x17[0, 0] = complex(float(df.iloc[0]['D3 (180°)']), 0)

print("17x17傅里叶系数矩阵：")
print(fourier_matrix_17x17)

# 创建128x128的傅里叶系数矩阵
image_size = 128
fourier_matrix_128x128 = np.zeros((image_size, image_size), dtype=complex)

# 填充128x128矩阵，只在fx和fy为0-16的位置设置值
for fx in range(17):
    for fy in range(17):
        # 交换fx和fy的顺序，解决图像翻转问题
        fourier_matrix_128x128[fy, fx] = fourier_matrix_17x17[fy, fx]

# 进行逆傅里叶变换
reconstructed_image = np.fft.ifft2(fourier_matrix_128x128)

# 翻转图像以纠正左右反转问题
reconstructed_image = np.fliplr(reconstructed_image)

# 获取实部作为重构的图像
reconstructed_image_real = np.real(reconstructed_image)

# 归一化到0-255范围
min_val = np.min(reconstructed_image_real)
max_val = np.max(reconstructed_image_real)
reconstructed_image_normalized = ((reconstructed_image_real - min_val) / (max_val - min_val)) * 255
reconstructed_image_normalized = reconstructed_image_normalized.astype(np.uint8)

print("\n重构的128x128图像数据范围：")
print(f"最小值: {np.min(reconstructed_image_normalized)}")
print(f"最大值: {np.max(reconstructed_image_normalized)}")
print(f"平均值: {np.mean(reconstructed_image_normalized)}")

# 保存图像
output_image_path = r"D:\桌面\毕设\data[16,16]\数据16的重构.png"
img = Image.fromarray(reconstructed_image_normalized, mode='L')
img.save(output_image_path)

print(f"\n重构完成！图像已保存到 {output_image_path}")

# 计算MSE和PSNR（需要原始图像数据）
print("\n注意：要计算MSE和PSNR，需要原始128x128图像的像素数据。")
print("如果您能提供原始图像的像素数据，我可以帮您计算这些指标。")

# 保存傅里叶系数幅度图（使用PIL）
fourier_amplitude = np.abs(fourier_matrix_17x17)
# 归一化到0-255
min_amp = np.min(fourier_amplitude)
max_amp = np.max(fourier_amplitude)
if max_amp > min_amp:
    fourier_amplitude_normalized = ((fourier_amplitude - min_amp) / (max_amp - min_amp)) * 255
else:
    fourier_amplitude_normalized = np.zeros_like(fourier_amplitude)
fourier_amplitude_normalized = fourier_amplitude_normalized.astype(np.uint8)

# 创建放大的傅里叶系数图像（每个像素放大10倍）
amplified_size = 17 * 10
amplified_fft = np.zeros((amplified_size, amplified_size), dtype=np.uint8)
for i in range(17):
    for j in range(17):
        amplified_fft[i*10:(i+1)*10, j*10:(j+1)*10] = fourier_amplitude_normalized[i, j]

output_fft_path = r"D:\桌面\毕设\data[16,16]\数据15的傅里叶系数幅度.png"
fft_img = Image.fromarray(amplified_fft, mode='L')
fft_img.save(output_fft_path)

print(f"傅里叶系数幅度图已保存到 {output_fft_path}")
