import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
# 新增的库：从scikit-image中导入TV降噪函数
from skimage.restoration import denoise_tv_chambolle
from pathlib import Path
from tqdm import tqdm  # 用于显示迭代进度条

# ==================== 参数配置 (压缩感知版) ====================
# --- 输入文件 ---
data_number = "4"
data_dir = Path(r"D:\桌面\毕设\data（压缩感知）\40%")
measurement_file = data_dir / f"{data_number}.tdms"
sampling_mask_file = data_dir / "sampling_matrix_40p.npy"
original_image_path = Path(r"D:\桌面\毕设\picture\测试图案\number1_128x128.png")

# --- 输出文件 ---
output_excel_file = data_dir / f"数据{data_number}处理结果.xlsx"
# 我们将保存两种重构图像，以作对比
output_image_zerofill_path = data_dir / f"数据{data_number}处理重构_零填充.png"
output_image_cs_path = data_dir / f"数据{data_number}处理重构_CS-TV.png"
output_comparison_path = data_dir / f"数据{data_number}处理对比_CS-TV.png"  # 最终对比图用CS结果

# --- 频率和图像尺寸配置 ---
FX_RANGE = (-64, 63)
FY_RANGE = (0, 63)
IMAGE_SIZE = 128

# ==================== 新增：CS重建算法配置 ====================
# --- 迭代重建参数 ---
# 迭代次数：增加次数可以提高质量，但会增加计算时间。通常50-100次足够。
CS_ITERATIONS = 70
# 总变分权重 (TV Weight)：这是最重要的参数！
# 它平衡了“数据一致性”和“图像稀疏性”。
# 值越大，图像越平滑（可能丢失细节）；值越小，越接近零填充结果（可能噪声更多）。
# 建议从 0.1 开始尝试，可以在 [0.05, 0.5] 范围内调整以获得最佳效果。
CS_TV_WEIGHT = 0.1


# =================================================================

def read_measurement_data(filepath):
    """通用数据读取函数，支持 tdms, txt, csv。"""
    # (此函数与您提供的代码相同，因此省略以保持简洁)
    # ...
    # 假设该函数已正确定义
    data = []
    filepath = Path(filepath)
    try:
        from nptdms import TdmsFile
        with TdmsFile.open(filepath) as tdms_file:
            group = tdms_file.groups()[0]
            channel = group.channels()[0]
            data = channel[:].tolist()
        return data
    except Exception:
        pass
    try:
        data = np.loadtxt(filepath).tolist()
        return data
    except Exception:
        pass
    try:
        data = pd.read_csv(filepath, header=None).iloc[:, 0].tolist()
        return data
    except Exception as e:
        raise IOError(f"所有格式读取失败: {filepath}. 错误: {e}")


# ==================== 新增：真正的CS重建函数 ====================
def reconstruct_cs_tv(fourier_coeffs, image_size, iterations, tv_weight):
    """
    使用基于总变分(TV)最小化的迭代算法重建图像。

    参数:
    - fourier_coeffs: 包含测量到的傅里叶系数的字典 {(fx, fy): coeff}。
    - image_size: 重建图像的边长。
    - iterations: 迭代次数。
    - tv_weight: TV去噪的权重。

    返回:
    - 重建后的图像 (numpy array)。
    """
    print("\n--- 开始执行CS-TV迭代重建 ---")
    print(f"迭代次数: {iterations}, TV权重: {tv_weight}")

    # 1. 创建k空间测量矩阵和采样掩码
    k_space_measured = np.zeros((image_size, image_size), dtype=complex)
    sampling_mask = np.zeros((image_size, image_size), dtype=bool)

    for (fx, fy), coeff in fourier_coeffs.items():
        # k空间中心为(size/2, size/2)
        kx, ky = (fx + image_size // 2) % image_size, (fy + image_size // 2) % image_size
        if 0 <= kx < image_size and 0 <= ky < image_size:
            k_space_measured[ky, kx] = coeff
            sampling_mask[ky, kx] = True
        # 共轭对称点
        if fx != 0 or fy != 0:
            kx_conj, ky_conj = (-fx + image_size // 2) % image_size, (-fy + image_size // 2) % image_size
            if 0 <= kx_conj < image_size and 0 <= ky_conj < image_size:
                k_space_measured[ky, kx_conj] = np.conj(coeff)
                sampling_mask[ky, kx_conj] = True

    # 2. 初始猜测：使用零填充的IFFT结果作为起点
    # (这就是您之前的重建方法)
    x = np.fft.ifft2(np.fft.ifftshift(k_space_measured))

    # 3. 迭代过程
    for i in tqdm(range(iterations), desc="CS-TV 迭代"):
        # 步骤a: 强制数据一致性
        k_current = np.fft.fftshift(np.fft.fft2(x))
        k_current[sampling_mask] = k_space_measured[sampling_mask]  # 将测量点的值替换回真实测量值
        x_corrected = np.fft.ifft2(np.fft.ifftshift(k_current))

        # 步骤b: 强制稀疏性 (TV去噪)
        # denoise_tv_chambolle只处理实数部分，这符合我们的目标
        x = denoise_tv_chambolle(np.real(x_corrected), weight=tv_weight)

    print("--- CS-TV迭代重建完成 ---")
    return x


# =================================================================

def main():
    # --- 步骤 1 & 2 & 3: 读取数据和计算傅里叶系数 ---
    # (这部分代码与您提供的完全相同，因此省略，假设它们已正确执行)
    print("正在执行步骤 1, 2, 3: 读取数据、确定频点、计算傅里叶系数...")
    # ... (此处应为您的完整代码)
    # 假设执行完后，我们得到了 `fourier_coeffs` 这个字典
    raw_data = read_measurement_data(measurement_file)
    averages = []
    if raw_data:
        for i in range(0, len(raw_data), 10):
            group = sorted(raw_data[i:i + 10])
            filtered_group = group[2:-2] if len(group) >= 6 else group
            if filtered_group:
                averages.append(sum(filtered_group) / len(filtered_group))
    sampling_mask_np = np.load(sampling_mask_file)
    sampled_freq_points = []
    fy_dim, fx_dim = sampling_mask_np.shape
    mask_fx_offset = -FX_RANGE[0]
    for fy in range(FY_RANGE[0], FY_RANGE[1] + 1):
        for fx in range(FX_RANGE[0], FX_RANGE[1] + 1):
            if sampling_mask_np[fy, fx + mask_fx_offset] == 1:
                sampled_freq_points.append((fx, fy))
    sampled_freq_points.sort(key=lambda p: (p[0], p[1]))
    actual_points = min(len(averages) // 4, len(sampled_freq_points))
    sampled_freq_points = sampled_freq_points[:actual_points]
    fourier_coeffs = {}
    output_data = []
    for i, (fx, fy) in enumerate(sampled_freq_points):
        d1, d2, d3, d4 = averages[i * 4: i * 4 + 4]
        d1_d3, d2_d4 = d1 - d3, d2 - d4
        coeff = d1_d3 - 1j * d2_d4
        if fx == 0 and fy == 0: coeff = (d1_d3 + d2_d4) / 2
        fourier_coeffs[(fx, fy)] = coeff
    print("步骤 1, 2, 3 完成。")
    print("-" * 30)

    # --- 步骤 4: 重建图像 (新旧方法对比) ---
    print("--- 步骤 4: 重建图像 ---")

    # 方法一：零填充 (您原来的方法，作为对比)
    k_space_zerofill = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=complex)
    for (fx, fy), coeff in fourier_coeffs.items():
        kx, ky = (fx + IMAGE_SIZE // 2) % IMAGE_SIZE, (fy + IMAGE_SIZE // 2) % IMAGE_SIZE
        if 0 <= kx < IMAGE_SIZE and 0 <= ky < IMAGE_SIZE:
            k_space_zerofill[ky, kx] = coeff
        if fx != 0 or fy != 0:
            kx_conj, ky_conj = (-fx + IMAGE_SIZE // 2) % IMAGE_SIZE, (-fy + IMAGE_SIZE // 2) % IMAGE_SIZE
            if 0 <= kx_conj < IMAGE_SIZE and 0 <= ky_conj < IMAGE_SIZE:
                k_space_zerofill[ky, kx_conj] = np.conj(coeff)

    reconstructed_image_zerofill = np.real(np.fft.ifft2(np.fft.ifftshift(k_space_zerofill)))

    # 方法二：真正的压缩感知重建 (CS-TV)
    reconstructed_image_cs = reconstruct_cs_tv(
        fourier_coeffs,
        IMAGE_SIZE,
        CS_ITERATIONS,
        CS_TV_WEIGHT
    )

    # --- 图像后处理 (归一化和翻转) ---
    # 对CS重建结果进行处理
    min_val, max_val = np.min(reconstructed_image_cs), np.max(reconstructed_image_cs)
    print(f"CS重构图像亮度范围: 最小值={min_val:.2f}, 最大值={max_val:.2f}")
    if max_val > min_val:
        reconstructed_image_norm = 255 * (reconstructed_image_cs - min_val) / (max_val - min_val)
    else:
        reconstructed_image_norm = np.zeros_like(reconstructed_image_cs)

    reconstructed_image_uint8 = reconstructed_image_norm.astype(np.uint8)
    reconstructed_image_flipped = np.fliplr(reconstructed_image_uint8)  # 最终用于评估的图像
    print("已对CS重构图像进行归一化和左右翻转")

    # 保存CS重建的图像
    Image.fromarray(reconstructed_image_flipped).save(output_image_cs_path)
    print(f"CS-TV重构图像已保存到: {output_image_cs_path}")

    # (可选) 同时保存零填充的图像，方便对比
    zf_min, zf_max = np.min(reconstructed_image_zerofill), np.max(reconstructed_image_zerofill)
    if zf_max > zf_min:
        zf_norm = 255 * (reconstructed_image_zerofill - zf_min) / (zf_max - zf_min)
        Image.fromarray(np.fliplr(zf_norm.astype(np.uint8))).save(output_image_zerofill_path)
        print(f"零填充重构图像已保存到: {output_image_zerofill_path}")

    print("-" * 30)

    # --- 步骤 5: 性能评估与对比图生成 (使用CS结果) ---
    print("--- 步骤 5: 图像质量评估与对比图生成 (基于CS-TV结果) ---")
    try:
        original_img = Image.open(original_image_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        original_array = np.array(original_img)

        # 计算MSE, PSNR, SSIM（基于CS翻转后的图像）
        mse = np.mean((original_array - reconstructed_image_flipped) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ssim_val = ssim(original_array, reconstructed_image_flipped, data_range=255)

        print(f"CS-TV 重建结果评估: MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")

        # 创建对比图
        comparison_img = Image.new('RGB', (IMAGE_SIZE * 2 + 60, IMAGE_SIZE + 80), (255, 255, 255))
        draw = ImageDraw.Draw(comparison_img)
        comparison_img.paste(original_img, (20, 40))
        comparison_img.paste(Image.fromarray(reconstructed_image_flipped), (IMAGE_SIZE + 40, 40))

        try:
            font = ImageFont.truetype("msyh.ttc", 16)
        except IOError:
            font = ImageFont.load_default()

        draw.text((20, 10), "原始图像", fill="black", font=font)
        draw.text((IMAGE_SIZE + 40, 10), f"CS-TV 重构 ({data_number})", fill="black", font=font)
        info_text = f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}"
        draw.text((2, IMAGE_SIZE + 50), info_text, fill="black", font=font)

        comparison_img.save(output_comparison_path)
        print(f"对比图像已保存到: {output_comparison_path}")

    except Exception as e:
        print(f"生成对比图时发生错误: {e}")

    print("\n所有处理完成！")


if __name__ == '__main__':
    main()