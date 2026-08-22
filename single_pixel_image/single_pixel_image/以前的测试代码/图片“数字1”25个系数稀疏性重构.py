"""
对输入excel表格数据进行稀疏重构
excel表格位置在289行，输出位置在290行


"""
import numpy as np
import pandas as pd
from PIL import Image
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
plt.rc("font",family='YouYuan')

def parse_fourier_data_from_excel(excel_path):
    """
    从Excel文件中解析傅里叶系数数据

    参数:
        excel_path: Excel文件路径

    返回:
        dict: 包含傅里叶系数、幅度和相位的字典
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        print(f"成功读取Excel文件: {excel_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 解析数据
        fourier_data = {}

        for idx, row in df.iterrows():
            # 从"频率点"列解析fx和fy
            freq_str = str(row['频率点'])

            # 使用正则表达式提取fx和fy值
            fx_match = re.search(r'fx=([0-9\-]+)', freq_str)
            fy_match = re.search(r'fy=([0-9\-]+)', freq_str)

            if fx_match and fy_match:
                fx = int(fx_match.group(1))
                fy = int(fy_match.group(1))

                # 解析复数
                complex_str = str(row['复数'])
                # 提取实部和虚部
                complex_match = re.search(r'([\+\-]?\d+\.\d+)([\+\-]\d+\.\d+)j', complex_str)
                if complex_match:
                    real_part = float(complex_match.group(1))
                    imag_part = float(complex_match.group(2))
                else:
                    # 尝试其他格式
                    real_part = float(row['复数(实部)']) if '复数(实部)' in df.columns else 0
                    imag_part = float(row['复数(虚部)']) if '复数(虚部)' in df.columns else 0

                magnitude = float(row['幅度'])
                phase = float(row['相位'])

                # 存储数据
                fourier_data[(fx, fy)] = {
                    'complex': complex(real_part, imag_part),
                    'magnitude': magnitude,
                    'phase': phase
                }
            else:
                print(f"警告: 无法解析行{idx}的频率点: {freq_str}")

        print(f"成功解析了 {len(fourier_data)} 个频率点")
        return fourier_data

    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None


def reconstruct_image_from_fourier_coeff(fourier_data, image_size=128):
    """
    从傅里叶系数重构图像

    参数:
        fourier_data: 傅里叶系数字典
        image_size: 重构图像尺寸，默认128×128

    返回:
        np.array: 重构的图像数组
    """
    # 创建空的傅里叶频谱数组
    spectrum = np.zeros((image_size, image_size), dtype=complex)

    # 获取中心位置
    crow, ccol = image_size // 2, image_size // 2

    # 填充傅里叶系数
    for (fx, fy), coeff_data in fourier_data.items():
        # 计算在频谱中的位置
        row = crow + fy
        col = ccol + fx

        # 检查索引是否在有效范围内
        if 0 <= row < image_size and 0 <= col < image_size:
            spectrum[row, col] = coeff_data['complex']
        else:
            print(f"警告: 频率点(fx={fx}, fy={fy})超出图像范围")

    # 应用共轭对称性（对于实值图像）
    # 对于每个非零频率，添加其共轭对称点
    for (fx, fy), coeff_data in fourier_data.items():
        # 跳过DC分量(0,0)和已经处理过的点
        if fx == 0 and fy == 0:
            continue

        # 计算共轭对称位置
        conj_row = crow - fy
        conj_col = ccol - fx

        # 检查索引是否在有效范围内
        if 0 <= conj_row < image_size and 0 <= conj_col < image_size:
            # 如果该位置还没有被填充，则填充其共轭
            if spectrum[conj_row, conj_col] == 0:
                spectrum[conj_row, conj_col] = np.conj(coeff_data['complex'])

    # 将零频移回原始位置
    spectrum_shifted = np.fft.ifftshift(spectrum)

    # 执行逆傅里叶变换
    reconstructed = np.fft.ifft2(spectrum_shifted)

    # 取实部（由于数值误差可能会有很小的虚部）
    reconstructed_real = np.real(reconstructed)

    # 归一化到0-255范围
    reconstructed_normalized = 255 * (reconstructed_real - reconstructed_real.min()) / (
                reconstructed_real.max() - reconstructed_real.min())

    return reconstructed_normalized.astype(np.uint8)


def reconstruct_with_amplitude_phase(fourier_data, image_size=128):
    """
    使用幅度和相位信息重构图像

    参数:
        fourier_data: 傅里叶系数字典
        image_size: 重构图像尺寸

    返回:
        np.array: 重构的图像数组
    """
    # 创建空的傅里叶频谱数组
    spectrum = np.zeros((image_size, image_size), dtype=complex)

    # 获取中心位置
    crow, ccol = image_size // 2, image_size // 2

    # 从幅度和相位重建复数
    for (fx, fy), coeff_data in fourier_data.items():
        magnitude = coeff_data['magnitude']
        phase = coeff_data['phase']

        # 计算复数
        coeff = magnitude * np.exp(1j * phase)

        # 计算在频谱中的位置
        row = crow + fy
        col = ccol + fx

        # 检查索引是否在有效范围内
        if 0 <= row < image_size and 0 <= col < image_size:
            spectrum[row, col] = coeff

    # 应用共轭对称性
    for (fx, fy), coeff_data in fourier_data.items():
        if fx == 0 and fy == 0:
            continue

        magnitude = coeff_data['magnitude']
        phase = coeff_data['phase']
        coeff = magnitude * np.exp(1j * phase)

        # 计算共轭对称位置
        conj_row = crow - fy
        conj_col = ccol - fx

        # 检查索引是否在有效范围内
        if 0 <= conj_row < image_size and 0 <= conj_col < image_size:
            if spectrum[conj_row, conj_col] == 0:
                # 共轭对称
                spectrum[conj_row, conj_col] = np.conj(coeff)

    # 将零频移回原始位置
    spectrum_shifted = np.fft.ifftshift(spectrum)

    # 执行逆傅里叶变换
    reconstructed = np.fft.ifft2(spectrum_shifted)

    # 取实部
    reconstructed_real = np.real(reconstructed)

    # 归一化到0-255范围
    reconstructed_normalized = 255 * (reconstructed_real - reconstructed_real.min()) / (
                reconstructed_real.max() - reconstructed_real.min())

    return reconstructed_normalized.astype(np.uint8)


def visualize_reconstruction(original_image, reconstructed_image, save_dir, image_name):
    """
    可视化原始图像和重构图像

    参数:
        original_image: 原始图像数组
        reconstructed_image: 重构图像数组
        save_dir: 保存目录
        image_name: 图像名称
    """
    # 创建可视化图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 重构图像
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title('重构图像')
    axes[1].axis('off')

    # 差异图像
    diff = np.abs(original_image.astype(float) - reconstructed_image.astype(float))
    im_diff = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('差异图像')
    axes[2].axis('off')

    # 添加颜色条
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)

    # 计算差异统计
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

    # 添加统计信息
    fig.text(0.5, 0.02, f'MSE: {mse:.2f}, PSNR: {psnr:.2f} dB',
             ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f"{image_name}_reconstruction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"重构结果已保存到: {save_path}")
    print(f"差异统计: MSE = {mse:.2f}, PSNR = {psnr:.2f} dB")

    return save_path


def save_reconstructed_image(image_array, save_dir, image_name):
    """
    保存重构的图像

    参数:
        image_array: 图像数组
        save_dir: 保存目录
        image_name: 图像名称
    """
    # 转换为PIL图像
    image = Image.fromarray(image_array, mode='L')

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存图像
    save_path = os.path.join(save_dir, f"{image_name}_reconstructed.bmp")
    image.save(save_path)

    print(f"重构图像已保存到: {save_path}")
    return save_path


def main():
    """主函数"""
    # 配置参数
    EXCEL_DIR = r"D:\桌面\毕设\picture\傅里叶基底1bit（1024×1024）"
    OUTPUT_DIR = r"D:\桌面\毕设\picture\重构图像"

    # 查找Excel文件
    excel_files = list(Path(EXCEL_DIR).glob("*_fourier_coeff.xlsx"))

    if not excel_files:
        print(f"在目录 {EXCEL_DIR} 中未找到Excel文件")
        return

    print(f"找到 {len(excel_files)} 个Excel文件")

    for excel_path in excel_files:
        print(f"\n处理文件: {excel_path.name}")

        # 从Excel解析数据
        fourier_data = parse_fourier_data_from_excel(excel_path)

        if fourier_data is None or len(fourier_data) == 0:
            print("无法解析傅里叶系数数据，跳过此文件")
            continue

        # 检查是否有原始图像可用于比较
        base_name = excel_path.name.replace("_fourier_coeff.xlsx", "")
        original_image_path = excel_path.parent.parent / "测试图案" / f"{base_name}.bmp"

        original_image = None
        if original_image_path.exists():
            try:
                original_image = np.array(Image.open(original_image_path).convert('L'))
                print(f"找到原始图像: {original_image_path}")
            except Exception as e:
                print(f"无法读取原始图像: {e}")

        # 方法1: 从复数系数重构
        print("\n方法1: 从复数系数重构图像")
        reconstructed_complex = reconstruct_image_from_fourier_coeff(fourier_data, image_size=128)

        # 方法2: 从幅度和相位重构
        print("\n方法2: 从幅度和相位重构图像")
        reconstructed_amp_phase = reconstruct_with_amplitude_phase(fourier_data, image_size=128)

        # 保存重构图像
        save_reconstructed_image(reconstructed_complex, OUTPUT_DIR, f"{base_name}_complex")
        save_reconstructed_image(reconstructed_amp_phase, OUTPUT_DIR, f"{base_name}_amp_phase")

        # 如果有原始图像，进行可视化比较
        if original_image is not None:
            print(f"\n与原始图像进行比较...")

            # 调整原始图像大小以匹配重构图像
            if original_image.shape != (128, 128):
                original_image_resized = np.array(Image.fromarray(original_image).resize((128, 128), Image.LANCZOS))
            else:
                original_image_resized = original_image

            # 可视化比较
            visualize_reconstruction(original_image_resized, reconstructed_complex, OUTPUT_DIR, f"{base_name}_complex")
            visualize_reconstruction(original_image_resized, reconstructed_amp_phase, OUTPUT_DIR,
                                     f"{base_name}_amp_phase")

        # 打印频率系数统计
        print(f"\n频率系数统计:")
        print(f"  频率点总数: {len(fourier_data)}")

        # 按fx和fy分组统计
        fx_values = sorted(set(fx for fx, _ in fourier_data.keys()))
        fy_values = sorted(set(fy for _, fy in fourier_data.keys()))
        print(f"  fx范围: {min(fx_values)} 到 {max(fx_values)}")
        print(f"  fy范围: {min(fy_values)} 到 {max(fy_values)}")

        # 计算幅度统计
        magnitudes = [data['magnitude'] for data in fourier_data.values()]
        print(f"  幅度范围: {min(magnitudes):.2e} 到 {max(magnitudes):.2e}")
        print(f"  平均幅度: {np.mean(magnitudes):.2e}")

        # 查找主要频率成分
        sorted_by_magnitude = sorted(fourier_data.items(), key=lambda x: x[1]['magnitude'], reverse=True)
        print(f"\n前5个主要频率成分:")
        for (fx, fy), data in sorted_by_magnitude[:5]:
            print(f"  (fx={fx}, fy={fy}): 幅度={data['magnitude']:.2e}, 相位={data['phase']:.3f} rad")


if __name__ == "__main__":
    main()