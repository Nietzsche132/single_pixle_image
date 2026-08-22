import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from pathlib import Path
import time

# ==================== 改进的压缩感知参数配置 ====================
# --- 输入文件 ---
data_number = "6"
data_dir = Path(r"D:\桌面\毕设\data（压缩感知）\40%")
measurement_file = data_dir / f"{data_number}.tdms"
sampling_mask_file = data_dir / "sampling_matrix_40p.npy"
original_image_path = Path(r"D:\桌面\毕设\picture\测试图案\number2_128x128.png")

# --- 输出文件 ---
output_image_path = data_dir / f"数据{data_number}处理结果_加权.png"
output_comparison_path = data_dir / f"数据{data_number}处理对比_加权.png"
output_excel_file = data_dir / f"数据{data_number}处理结果_加权.xlsx"

# --- 频率和图像尺寸配置 ---
FX_RANGE = (-64, 63)
FY_RANGE = (0, 63)
IMAGE_SIZE = 128

# --- 重建参数 ---
TV_WEIGHT = 0.145


# =================================================================

def read_measurement_data(filepath):
    """读取测量数据"""
    data = []
    filepath = Path(filepath)
    print(f"开始读取测量数据: {filepath}")

    try:
        from nptdms import TdmsFile
        with TdmsFile.open(filepath) as tdms_file:
            group = tdms_file.groups()[0]
            channel = group.channels()[0]
            data = channel[:].tolist()
        print(f"成功读取TDMS文件，数据点数: {len(data)}")
        return data
    except Exception:
        print("读取TDMS失败，尝试其他格式...")
        try:
            data = np.loadtxt(filepath).tolist()
            print(f"成功读取文本文件，数据点数: {len(data)}")
            return data
        except Exception:
            try:
                data = pd.read_csv(filepath, header=None).iloc[:, 0].tolist()
                print(f"成功读取CSV文件，数据点数: {len(data)}")
                return data
            except Exception as e:
                raise IOError(f"所有格式读取失败: {filepath}. 错误: {e}")


def weighted_reconstruct(fourier_dict, mask, image_size=128, tv_weight=0.1):
    """
    加权全变分最小化重建
    """
    print(f"\n=== 加权全变分最小化重建 ===")
    print(f"图像尺寸: {image_size}x{image_size}")
    print(f"采样率: {np.sum(mask) / mask.size * 100:.2f}%")
    print(f"全变分权重: {tv_weight}")
    print("-" * 30)

    freq_full = np.zeros((image_size, image_size), dtype=complex)

    for (fx, fy), coeff in fourier_dict.items():
        kx = (fx + 64) % image_size
        ky = (fy + 64) % image_size
        freq_full[ky, kx] = coeff

        if fx != 0 or fy != 0:
            kx_conj = (-fx + 64) % image_size
            ky_conj = (-fy + 64) % image_size
            freq_full[ky_conj, kx_conj] = np.conj(coeff)

    print(f"频域填充完成，采样点数: {len(fourier_dict) * 2 - 1}")

    freq_shifted = np.fft.ifftshift(freq_full)
    initial_estimate = np.fft.ifft2(freq_shifted)
    initial_estimate = np.real(initial_estimate)

    print(f"初始估计完成，亮度范围: [{np.min(initial_estimate):.4f}, {np.max(initial_estimate):.4f}]")

    img_min, img_max = np.min(initial_estimate), np.max(initial_estimate)
    if img_max > img_min:
        img_normalized = (initial_estimate - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(initial_estimate)

    print(f"开始全变分优化...")

    try:
        img_denoised = denoise_tv_chambolle(img_normalized, weight=tv_weight)
        print(f"全变分优化完成")
    except Exception as e:
        print(f"全变分优化失败: {e}")
        img_denoised = img_normalized

    reconstructed = img_denoised * (img_max - img_min) + img_min

    print(f"重建完成，最终亮度范围: [{np.min(reconstructed):.4f}, {np.max(reconstructed):.4f}]")

    return reconstructed


def iterative_constraint_projection(fourier_dict, initial_image, mask, image_size=128, n_iter=10, tv_weight=0.05):
    """
    迭代约束投影重建
    """
    print(f"\n=== 迭代约束投影重建 ===")
    print(f"迭代次数: {n_iter}")
    print(f"全变分权重: {tv_weight}")
    print("-" * 30)

    img = initial_image.copy()

    for iteration in range(n_iter):
        freq = np.fft.fft2(img)
        freq_shifted = np.fft.fftshift(freq)

        freq_constrained = np.zeros_like(freq_shifted)
        for (fx, fy), coeff in fourier_dict.items():
            kx = (fx + 64) % image_size
            ky = (fy + 64) % image_size
            freq_constrained[ky, kx] = coeff

            if fx != 0 or fy != 0:
                kx_conj = (-fx + 64) % image_size
                ky_conj = (-fy + 64) % image_size
                freq_constrained[ky_conj, kx_conj] = np.conj(coeff)

        freq_constrained_shifted = np.fft.ifftshift(freq_constrained)
        img_constrained = np.fft.ifft2(freq_constrained_shifted)
        img_constrained = np.real(img_constrained)

        img_min, img_max = np.min(img_constrained), np.max(img_constrained)
        if img_max > img_min:
            img_normalized = (img_constrained - img_min) / (img_max - img_min)
        else:
            img_normalized = np.zeros_like(img_constrained)

        img_denoised = denoise_tv_chambolle(img_normalized, weight=tv_weight)

        img = img_denoised * (img_max - img_min) + img_min

        if (iteration + 1) % 5 == 0 or iteration == 0:
            mse = np.mean((img - img_constrained) ** 2)
            print(f"迭代 {iteration + 1}/{n_iter}: MSE={mse:.6f}")

    return img


def main():
    start_time = time.time()

    print("=" * 50)
    print("改进的压缩感知重建程序")
    print("=" * 50)

    print("\n--- 步骤 1: 读取并处理测量数据 ---")
    raw_data = read_measurement_data(measurement_file)

    averages = []
    if raw_data:
        for i in range(0, len(raw_data), 10):
            group = sorted(raw_data[i:i + 10])
            if len(group) >= 6:
                filtered_group = group[2:-2]
            else:
                filtered_group = group
            if filtered_group:
                averages.append(sum(filtered_group) / len(filtered_group))
    print(f"数据预处理完成，得到 {len(averages)} 个有效测量值。")
    print("-" * 30)

    print("--- 步骤 2: 加载采样矩阵 ---")
    sampling_mask = np.load(sampling_mask_file)
    print(f"采样矩阵形状: {sampling_mask.shape}")

    sampled_freq_points = []
    fy_dim, fx_dim = sampling_mask.shape
    mask_fx_offset = 64

    for fy in range(FY_RANGE[0], FY_RANGE[1] + 1):
        for fx in range(FX_RANGE[0], FX_RANGE[1] + 1):
            mask_row = fy
            mask_col = fx + mask_fx_offset
            if 0 <= mask_row < fy_dim and 0 <= mask_col < fx_dim:
                if sampling_mask[mask_row, mask_col] == 1:
                    sampled_freq_points.append((fx, fy))

    sampled_freq_points.sort(key=lambda p: (p[0], p[1]))
    num_sampled_points = len(sampled_freq_points)
    print(f"从掩模中解析出 {num_sampled_points} 个被测量的频点。")
    print("-" * 30)

    print("--- 步骤 3: 计算傅里叶系数 ---")
    fourier_coeffs = {}
    dc_component = None

    for i, (fx, fy) in enumerate(sampled_freq_points):
        start_idx = i * 4
        values = averages[start_idx: start_idx + 4]

        if len(values) == 4:
            d1, d2, d3, d4 = values
            d1_d3 = d1 - d3
            d2_d4 = d2 - d4

            if fx == 0 and fy == 0:
                dc_component = (d1_d3 + d2_d4) / 2
                coeff = dc_component
            else:
                coeff = d1_d3 - 1j * d2_d4

            fourier_coeffs[(fx, fy)] = coeff

    print(f"0频分量: {dc_component}")
    print(f"共计算 {len(fourier_coeffs)} 个傅里叶系数")
    print("-" * 30)

    print("--- 步骤 4: 构建频域掩模 ---")

    full_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)

    for (fx, fy), coeff in fourier_coeffs.items():
        kx = (fx + 64) % IMAGE_SIZE
        ky = (fy + 64) % IMAGE_SIZE
        full_mask[ky, kx] = 1

        if fx != 0 or fy != 0:
            kx_conj = (-fx + 64) % IMAGE_SIZE
            ky_conj = (-fy + 64) % IMAGE_SIZE
            full_mask[ky_conj, kx_conj] = 1

    print(f"频域掩模中1的数量: {np.sum(full_mask)}")
    print("-" * 30)

    print("--- 步骤 5: 加权全变分重建 ---")
    reconstructed_image_tv = weighted_reconstruct(
        fourier_coeffs,
        full_mask,
        image_size=IMAGE_SIZE,
        tv_weight=TV_WEIGHT
    )

    print("\n--- 步骤 6: 迭代约束投影 ---")
    reconstructed_image = iterative_constraint_projection(
        fourier_coeffs,
        reconstructed_image_tv,
        full_mask,
        image_size=IMAGE_SIZE,
        n_iter=20,
        tv_weight=TV_WEIGHT
    )

    min_val, max_val = np.min(reconstructed_image), np.max(reconstructed_image)
    print(f"\n重建图像亮度范围: [{min_val:.4f}, {max_val:.4f}]")

    if max_val > min_val:
        reconstructed_image_norm = 255 * (reconstructed_image - min_val) / (max_val - min_val)
    else:
        reconstructed_image_norm = np.zeros_like(reconstructed_image)

    reconstructed_image_uint8 = reconstructed_image_norm.astype(np.uint8)

    reconstructed_image_flipped = np.fliplr(reconstructed_image_uint8)
    print("已对重构图像进行左右翻转")

    Image.fromarray(reconstructed_image_flipped).save(output_image_path)
    print(f"重构图像已保存到: {output_image_path}")
    print("-" * 30)

    print("--- 步骤 7: 图像质量评估 ---")
    try:
        original_img = Image.open(original_image_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        original_array = np.array(original_img)

        mse = np.mean((original_array - reconstructed_image_flipped) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ssim_val = ssim(original_array, reconstructed_image_flipped, data_range=255)

        print(f"\n=== 重建质量指标 ===")
        print(f"MSE: {mse:.2f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")
        print("=" * 30)

        comparison_img = Image.new('RGB', (IMAGE_SIZE * 2 + 60, IMAGE_SIZE + 80), (255, 255, 255))
        draw = ImageDraw.Draw(comparison_img)

        comparison_img.paste(original_img, (20, 40))
        comparison_img.paste(Image.fromarray(reconstructed_image_flipped), (IMAGE_SIZE + 40, 40))

        try:
            font = ImageFont.truetype("msyh.ttc", 16)
        except IOError:
            font = ImageFont.load_default()

        draw.text((20, 10), "原始图像", fill="black", font=font)
        draw.text((IMAGE_SIZE + 40, 10), "加权TV重构", fill="black", font=font)
        info_text = f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}"
        draw.text((2, IMAGE_SIZE + 50), info_text, fill="black", font=font)

        comparison_img.save(output_comparison_path)
        print(f"对比图像已保存到: {output_comparison_path}")

    except Exception as e:
        print(f"生成对比图时发生错误: {e}")

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")
    print("\n所有处理完成！")


if __name__ == '__main__':
    main()