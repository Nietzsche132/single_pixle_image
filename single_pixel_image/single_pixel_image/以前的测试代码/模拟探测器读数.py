import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rc("font",family='YouYuan')
def generate_fourier_pattern(m, n, fx, fy, phase_deg):
    """
    生成傅里叶基底照明图案
    参数:
        m, n: 图像尺寸
        fx, fy: 频率坐标
        phase_deg: 相位角（度）
    返回:
        pattern: 生成的图案矩阵
    """
    # 创建坐标网格
    x = np.arange(n) / n  # 归一化到 [0, 1)
    y = np.arange(m) / m
    X, Y = np.meshgrid(x, y)

    # 将零点移到中心（fftshift效果）
    X = np.fft.fftshift(X)
    Y = np.fft.fftshift(Y)

    # 计算相位（弧度）
    phase_rad = np.deg2rad(phase_deg)

    # 生成图案: 1 + cos(2π(fx*x + fy*y) + φ)
    pattern = 1 + np.cos(2 * np.pi * (fx * X + fy * Y) + phase_rad)

    return pattern


# ==================== 主程序 ====================

# 1. 加载图像
image_path = r"D:\桌面\毕设\picture\测试图案\number1_128x128.bmp"
img = Image.open(image_path)

# 转换为灰度图并转为numpy数组（范围0-1）
if img.mode != 'L':
    img = img.convert('L')
img_array = np.array(img, dtype=np.float64) / 255.0

# 获取图像尺寸
m, n = img_array.shape
print(f"图像尺寸: {m} × {n}")

# 2. 定义参数
fx = 1
fy = 1
phases = [0, 90, 180, 270]  # 四个相位角

# 3. 存储读数的列表
detector_readings = []

# 4. 对每个相位进行模拟测量
for phase in phases:
    # 生成照明图案
    pattern = generate_fourier_pattern(m, n, fx, fy, phase)

    # 模拟单像素探测器读数：sum(sum(pattern .* image))
    reading = np.sum(pattern * img_array)

    detector_readings.append(reading)

    # 显示图案（可选）
    plt.figure(figsize=(6, 6))
    plt.imshow(pattern, cmap='gray')
    plt.title(f'Fourier Pattern\nfx={fx}, fy={fy}, Phase={phase}°')
    plt.axis('off')
    plt.show()

# 5. 显示四个读数
print("\n" + "=" * 40)
print("模拟探测器读数结果:")
print("=" * 40)
for i, (phase, reading) in enumerate(zip(phases, detector_readings)):
    print(f"相位 {phase:3d}°   →   读数 = {reading:.6f}")

# 6. 可视化读数对比
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(phases)), detector_readings,
               color=['red', 'green', 'blue', 'orange'])
plt.xlabel('相位')
plt.ylabel('探测器读数')
plt.title(f'模拟单像素探测器读数 (fx={fx}, fy={fy})')
plt.xticks(range(len(phases)), [f'{p}°' for p in phases])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在柱状图上显示数值
for bar, reading in zip(bars, detector_readings):
    height = bar.get_height()
    plt.annotate(f'{reading:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.tight_layout()
plt.show()