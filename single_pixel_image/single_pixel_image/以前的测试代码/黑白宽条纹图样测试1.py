import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft2
import cv2
import os
import struct

plt.rc("font",family='YouYuan')
def generate_fourier_basis(M, N, u, v, phi):
    """
    生成傅里叶基底图案（傅里叶域法）
    针对1920×1080分辨率的优化实现

    参数:
        M, N: 图像分辨率（高度，宽度）
        u, v: 空间频率
        phi: 初相位（弧度）

    返回:
        P: M×N的矩阵，表示傅里叶基底图案，灰度值归一化到[0,1]
    """
    # 创建M×N的全零复矩阵，表示傅里叶谱
    F = np.zeros((M, N), dtype=complex)

    # 设置共轭对称点以确保生成的图案为纯实数
    u = int(np.clip(u, 0, N - 1))
    v = int(np.clip(v, 0, M - 1))

    # 在正频率位置设置脉冲
    F[v, u] = np.exp(1j * phi)
    # 在对应的负频率位置设置共轭对称点
    if u != 0 or v != 0:  # 避免在直流分量重复设置
        F[-v, -u] = np.exp(-1j * phi)

    # 对傅里叶谱进行二维逆傅里叶变换，并取实部
    P_temp = ifft2(F) * M * N
    P_real = np.real(P_temp)

    # 归一化图案到[0,1]范围
    min_val = np.min(P_real)
    max_val = np.max(P_real)
    P = (P_real - min_val) / (max_val - min_val)

    return P


def floyd_steinberg_dithering(image):
    """
    Floyd-Steinberg误差扩散抖动算法
    将灰度图像转换为二值图像

    参数:
        image: 输入灰度图像，值范围[0,1]或[0,255]

    返回:
        dithered_image: 二值图像，值为0或255
    """
    # 确保图像是浮点类型且范围在[0,255]
    if image.max() <= 1.0:
        image_float = image * 255.0
    else:
        image_float = image.astype(np.float64)

    # 创建输出图像的副本
    output = image_float.copy()
    height, width = output.shape

    # Floyd-Steinberg误差扩散系数
    for y in range(height):
        for x in range(width):
            old_pixel = output[y, x]
            new_pixel = 0 if old_pixel < 128 else 255
            output[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # 误差扩散到相邻像素
            if x + 1 < width:
                output[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                output[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < height:
                output[y + 1, x] += quant_error * 5 / 16
            if x + 1 < width and y + 1 < height:
                output[y + 1, x + 1] += quant_error * 1 / 16

    # 转换为uint8类型
    dithered_image = np.clip(output, 0, 255).astype(np.uint8)
    return dithered_image


def save_as_1bit_bmp(image, filename):
    """
    将二值图像保存为1位BMP文件
    参考BMP文件格式规范实现真正的1位位图保存

    参数:
        image: 二值图像，值应为0或255
        filename: 保存的文件路径
    """
    height, width = image.shape

    # 确保图像是二值的（0和255）
    binary_image = image.copy()
    if binary_image.max() <= 1.0:
        binary_image = (binary_image * 255).astype(np.uint8)

    # 转换为真正的二值（0和1）
    bit_image = (binary_image > 128).astype(np.uint8)

    # 计算行对齐（每行必须对齐到4字节）
    row_size = ((width + 31) // 32) * 4  # 每行字节数，对齐到4字节
    data_size = row_size * height

    # BMP文件头（14字节）
    file_header = bytearray(14)
    file_header[0:2] = b'BM'  # 文件类型
    file_size = 14 + 40 + 8 + data_size  # 文件头+信息头+调色板+数据
    struct.pack_into('<I', file_header, 2, file_size)  # 文件大小
    struct.pack_into('<I', file_header, 10, 14 + 40 + 8)  # 数据偏移量

    # BMP信息头（40字节）
    info_header = bytearray(40)
    struct.pack_into('<I', info_header, 0, 40)  # 信息头大小
    struct.pack_into('<I', info_header, 4, width)  # 图像宽度
    struct.pack_into('<I', info_header, 8, height)  # 图像高度
    struct.pack_into('<H', info_header, 12, 1)  # 颜色平面数
    struct.pack_into('<H', info_header, 14, 1)  # 每像素位数（1位）
    struct.pack_into('<I', info_header, 20, data_size)  # 图像数据大小

    # 调色板（8字节）- 1位需要2种颜色
    palette = bytearray(8)
    # 颜色0：黑色 (BGR)
    palette[0:3] = struct.pack('<BBB', 0, 0, 0)
    # 颜色1：白色 (BGR)
    palette[4:7] = struct.pack('<BBB', 255, 255, 255)

    # 生成位图数据
    bitmap_data = bytearray(data_size)

    for y in range(height):
        # BMP从下往上存储，所以需要反转y坐标
        bmp_y = height - 1 - y
        for x in range(width):
            # 获取像素值（0或1）
            pixel_value = bit_image[y, x]

            # 计算字节位置和位位置
            byte_index = bmp_y * row_size + x // 8
            bit_position = 7 - (x % 8)  # BMP中高位在前

            # 设置对应的位
            if pixel_value:
                bitmap_data[byte_index] |= (1 << bit_position)

    # 写入文件
    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write(info_header)
        f.write(palette)
        f.write(bitmap_data)


def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建

    参数:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")
    else:
        print(f"目录已存在: {directory}")


def main():
    # 设置参数
    M, N = 1080, 1920  # 图像分辨率：高度1080，宽度1920
    fx, fy = 0, 0  # 空间频率

    # 设置保存目录
    save_directory = r"D:\桌面\毕设\picture\傅里叶基底宽条纹（1920×1080）"

    # 确保保存目录存在
    ensure_directory_exists(save_directory)

    # 四步相移的相位值（弧度）
    phases_deg = [0, 90, 180, 270]
    phases_rad = [phase * np.pi / 180 for phase in phases_deg]
    phase_names = ['0°', '90°', '180°', '270°']

    print(f"生成傅里叶基底图案...")
    print(f"分辨率: {N}×{M}")
    print(f"空间频率: fx={fx}, fy={fy}")
    print(f"保存目录: {save_directory}")
    print(f"二值图像格式: 1位BMP")

    # 生成四张傅里叶基底图案
    original_patterns = []
    dithered_patterns = []

    for i, (phi_rad, phi_name) in enumerate(zip(phases_rad, phase_names)):
        # 生成傅里叶基底图案
        pattern = generate_fourier_basis(M, N, fx, fy, phi_rad)
        original_patterns.append(pattern)

        # 应用Floyd-Steinberg抖动
        dithered = floyd_steinberg_dithering(pattern)
        dithered_patterns.append(dithered)

        # 保存原始图像为8位灰度BMP格式
        original_uint8 = (pattern * 255).astype(np.uint8)
        original_filename = f"fourier_basis_phase_{phi_name.replace('°', 'deg')}_fx{fx}_fy{fy}_8bit.bmp"
        original_save_path = os.path.join(save_directory, original_filename)
        cv2.imwrite(original_save_path, original_uint8)

        # 保存二值图像为1位BMP格式
        dithered_filename = f"fourier_basis_dithered_phase_{phi_name.replace('°', 'deg')}_fx{fx}_fy{fy}_1bit.bmp"
        dithered_save_path = os.path.join(save_directory, dithered_filename)
        save_as_1bit_bmp(dithered, dithered_save_path)

        print(f"已生成并保存相位 {phi_name} 的图案:")
        print(f"  - 原始图案 (8位): {original_filename}")
        print(f"  - 抖动图案 (1位): {dithered_filename}")

    # 显示结果
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f'傅里叶基底图案 (fx={fx}, fy={fy}) 及 Floyd-Steinberg 抖动结果', fontsize=16)

    for i in range(4):
        # 显示原始傅里叶基底图案
        ax1 = axes[i, 0]
        im1 = ax1.imshow(original_patterns[i], cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'原始图案 - 相位 {phase_names[i]}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 显示抖动后的二值图像
        ax2 = axes[i, 1]
        im2 = ax2.imshow(dithered_patterns[i], cmap='gray', vmin=0, vmax=255)
        ax2.set_title(f'抖动后二值图像 - 相位 {phase_names[i]}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # 验证和统计信息
    print("\n=== 图案特性验证 ===")
    print(f"原始图案分辨率: {original_patterns[0].shape}")
    print(f"原始图案灰度范围: [{np.min(original_patterns[0]):.3f}, {np.max(original_patterns[0]):.3f}]")
    print(f"二值图像唯一值: {np.unique(dithered_patterns[0])}")

    # 计算文件大小对比
    print(f"\n=== 文件大小对比 ===")
    saved_files = [f for f in os.listdir(save_directory) if f.endswith('.bmp') and f'fx{fx}_fy{fy}' in f]
    for file in sorted(saved_files):
        file_path = os.path.join(save_directory, file)
        file_size = os.path.getsize(file_path)
        file_type = "1位二值" if "1bit" in file else "8位灰度"
        print(f"  - {file} ({file_type}): {file_size} bytes")

    # 显示保存确认信息
    print(f"\n=== 保存确认 ===")
    print(f"所有图像已成功保存到:")
    print(f"{save_directory}")
    print(f"共保存了 {len(phases_rad)} 个相位的图案")
    print(f"总文件数: {len(phases_rad) * 2} 个BMP文件 (8位灰度 + 1位二值)")


if __name__ == "__main__":
    main()