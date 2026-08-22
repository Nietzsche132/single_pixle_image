"""
求输入图像的傅里叶系数，其中求单个在120，121行修改fx和fy，求多个在124，125行修改fx和fy
修改输入图像的路径在117行
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rc("font", family='YouYuan')


def get_fourier_coefficient_at_fx_fy(image_path, fx=4, fy=4):
    """
    读取图像并获取指定频率(fx, fy)处的傅里叶系数

    参数:
        image_path: 图像文件路径
        fx: x方向频率 (默认4)
        fy: y方向频率 (默认4)

    返回:
        complex: 指定频率处的复傅里叶系数
    """
    # 1. 读取图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # 获取图像尺寸
    rows, cols = image_array.shape
    print(f"图像尺寸: {rows} x {cols}")

    # 2. 进行二维傅里叶变换
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)  # 将零频率移到中心

    # 3. 计算中心位置
    crow, ccol = rows // 2, cols // 2

    # 4. 计算指定频率处的索引位置
    # 注意：频率坐标是相对于图像中心的
    target_row = crow + fy
    target_col = ccol + fx

    # 检查索引是否在有效范围内
    if (0 <= target_row < rows) and (0 <= target_col < cols):
        coefficient = f_transform_shifted[target_row, target_col]

        # 计算幅度和相位
        magnitude = np.abs(coefficient)
        phase = np.angle(coefficient)

        print(f"\n在频率(fx={fx}, fy={fy})处的傅里叶系数:")
        print(f"  复数值: {coefficient}")
        print(f"  幅度: {magnitude:.6f}")
        print(f"  相位: {phase:.6f} rad")

        return coefficient
    else:
        print(f"错误: 频率(fx={fx}, fy={fy})超出图像频率范围")
        print(f"有效频率范围: fx=[{-ccol}, {cols - ccol - 1}], fy=[{-crow}, {rows - crow - 1}]")
        return None


def visualize_with_custom_frequencies(image_path, fx_values, fy_values):
    """
    可视化多个不同频率点的傅里叶系数

    参数:
        image_path: 图像文件路径
        fx_values: x方向频率列表或数组
        fy_values: y方向频率列表或数组
    """
    # 读取图像
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # 计算傅里叶变换
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(1 + np.abs(f_transform_shifted))

    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2

    # 创建图形
    plt.figure(figsize=(12, 5))

    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    # 显示幅度谱并标记所有频率点
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')

    # 标记每个频率点
    colors = ['ro', 'bo', 'go', 'yo', 'mo', 'co']
    for i, (fx, fy) in enumerate(zip(fx_values, fy_values)):
        color = colors[i % len(colors)]
        plt.plot(ccol + fx, crow + fy, color, markersize=8,
                 label=f'fx={fx}, fy={fy}')

    plt.title('频率谱 (标记多个点)')
    plt.legend(loc='upper right', fontsize=8)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # ==================== 配置参数区域 ====================
    # 在这里修改图像路径和频率值
    image_path = r"D:\桌面\毕设\picture\测试图案\number1_128x128.bmp"

    # 方法1: 单个频率点（直接修改这里的数值）
    single_fx = 4  # 修改为你想要的值
    single_fy = 4  # 修改为你想要的值

    # 方法2: 多个频率点（用于批量分析）
    multiple_fx = [3, 3, 3, 3]  # 可以添加更多频率
    multiple_fy = [1, 2, 3, 4]  # 与fx对应

    # ====================================================

    try:
        # 方式一：分析单个频率点
        print("=" * 50)
        print("分析单个频率点:")
        coefficient = get_fourier_coefficient_at_fx_fy(image_path, fx=single_fx, fy=single_fy)

        # 方式二：批量分析多个频率点
        print("\n" + "=" * 50)
        print("批量分析多个频率点:")
        coefficients = []
        for fx, fy in zip(multiple_fx, multiple_fy):
            coef = get_fourier_coefficient_at_fx_fy(image_path, fx=fx, fy=fy)
            coefficients.append(coef)

        # 可视化多个频率点
        visualize_with_custom_frequencies(image_path, multiple_fx, multiple_fy)

    except FileNotFoundError:
        print(f"错误: 找不到图像文件 {image_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")