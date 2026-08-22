import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# ==================== 参数配置 (压缩感知版) ====================
# --- 输入文件 ---
# 数据编号（只需修改这里）
data_number = "4"  # 例如，对应你的 "40%_1" 文件夹
# 数据目录
data_dir = Path(r"D:\桌面\毕设\data（压缩感知）\40%")
# 采集卡数据文件 (假设是 .tdms, .txt, 或 .csv)
measurement_file = data_dir / f"{data_number}.tdms"
# 采样矩阵（掩模）文件
sampling_mask_file = data_dir / "sampling_matrix_40p.npy"
# 原始对比图像
original_image_path = Path(r"D:\桌面\毕设\picture\测试图案\number1_128x128.png")

# --- 输出文件 ---
# 直接输出到数据目录，不创建子文件夹
# 输出Excel文件路径 (改回原来的文件名)
output_excel_file = data_dir / f"数据{data_number}处理结果.xlsx"
# 输出重构图像路径
output_image_path = data_dir / f"数据{data_number}处理重构.png"
# 输出对比图像路径
output_comparison_path = data_dir / f"数据{data_number}处理对比.png"

# --- 频率和图像尺寸配置 ---
# 注意：这里的范围必须与你的测量程序和掩模生成时的设置一致！
FX_RANGE = (-64, 63)
FY_RANGE = (0, 63)  # 整个fy范围（上下半平面）
IMAGE_SIZE = 128


# =================================================================

def read_measurement_data(filepath):
    """通用数据读取函数，支持 tdms, txt, csv。"""
    data = []
    filepath = Path(filepath)
    print(f"开始读取测量数据: {filepath}")

    # 优先尝试 nptdms
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

    # 尝试文本文件
    try:
        data = np.loadtxt(filepath).tolist()
        print(f"成功读取文本文件，数据点数: {len(data)}")
        return data
    except Exception:
        print("读取文本文件失败，尝试CSV格式...")

    # 尝试CSV文件
    try:
        data = pd.read_csv(filepath, header=None).iloc[:, 0].tolist()
        print(f"成功读取CSV文件，数据点数: {len(data)}")
        return data
    except Exception as e:
        raise IOError(f"所有格式读取失败，无法打开文件: {filepath}. 错误: {e}")


def main():
    # --- 步骤 1: 读取测量数据并预处理 ---
    print("--- 步骤 1: 读取并处理测量数据 ---")
    raw_data = read_measurement_data(measurement_file)

    averages = []
    # 从第一个数据开始，每10个为一组，去极值求平均
    if raw_data:
        for i in range(0, len(raw_data), 10):  # 从0开始，确保不丢失第一个数据点
            group = sorted(raw_data[i:i + 10])
            if len(group) >= 6:
                filtered_group = group[2:-2]
            else:
                filtered_group = group
            if filtered_group:
                averages.append(sum(filtered_group) / len(filtered_group))
    print(f"数据预处理完成，得到 {len(averages)} 个有效测量值。")
    print("-" * 30)

    # --- 步骤 2: 加载采样矩阵并确定测量的频点顺序 ---
    print("--- 步骤 2: 加载采样矩阵并确定频点顺序 ---")
    if not sampling_mask_file.exists():
        raise FileNotFoundError(f"采样矩阵文件不存在: {sampling_mask_file}")

    sampling_mask = np.load(sampling_mask_file)
    print(f"成功加载采样矩阵，形状: {sampling_mask.shape}")

    # 从掩模中找出被采样的频点 (fx, fy)
    # 注意：掩模的维度是 (fy_dim, fx_dim)，索引需要映射回 (fx, fy) 坐标
    sampled_freq_points = []
    fy_dim, fx_dim = sampling_mask.shape

    # 计算掩模的频率偏移
    mask_fx_min = -64  # 假设掩模的fx范围是[-64, 63]
    mask_fx_offset = -mask_fx_min

    print(f"采样矩阵形状: {sampling_mask.shape}")
    print(f"频率范围: fx={FX_RANGE}, fy={FY_RANGE}")
    print(f"掩模fx偏移: {mask_fx_offset}")

    for fy in range(FY_RANGE[0], FY_RANGE[1] + 1):
        for fx in range(FX_RANGE[0], FX_RANGE[1] + 1):
            mask_row = fy
            mask_col = fx + mask_fx_offset
            # 检查索引是否在有效范围内
            if 0 <= mask_row < fy_dim and 0 <= mask_col < fx_dim:
                # 检查掩模对应位置是否为1
                if sampling_mask[mask_row, mask_col] == 1:
                    sampled_freq_points.append((fx, fy))
            else:
                print(f"警告: 频点 ({fx}, {fy}) 对应的掩模索引 ({mask_row}, {mask_col}) 超出范围")

    #
    # !! 关键 !!
    # 这里的排序方式必须和你物理采集数据的顺序完全一致
    # 假设你也是按 fx 优先，然后 fy 的顺序扫描
    # 如果不是，请修改这里的排序逻辑
    #
    sampled_freq_points.sort(key=lambda p: (p[0], p[1]))

    num_sampled_points = len(sampled_freq_points)
    print(f"从掩模中解析出 {num_sampled_points} 个被测量的频点。")

    # 安全检查：测量值数量是否匹配？
    # 每个频点需要4个相位测量值
    expected_points = num_sampled_points * 4
    if len(averages) != expected_points:
        print(f"错误: 测量值数量 ({len(averages)}) 与预期数量 ({expected_points}) 不匹配！")
        print("请检查数据文件或掩模文件是否正确。")
        # 使用可用的数据，限制处理量
        actual_points = min(len(averages) // 4, num_sampled_points)
        if actual_points > 0:
            print(f"将使用前 {actual_points} 个频点进行处理。")
            sampled_freq_points = sampled_freq_points[:actual_points]
        else:
            print("错误: 数据不足，无法进行处理。")
            exit(1)
    print("-" * 30)

    # --- 步骤 3: 计算傅里叶系数并填充到Excel ---
    print("--- 步骤 3: 计算傅里叶系数 ---")
    output_data = []
    fourier_coeffs = {}  # 使用字典存储，键为(fx,fy)，值为复数系数
    normalized_coeffs = {}  # 存储归一化后的系数
    dc_component = None

    # 第一次遍历：计算所有傅里叶系数并找到0频分量
    temp_coeffs = {}
    for i, (fx, fy) in enumerate(sampled_freq_points):
        start_idx = i * 4
        values = averages[start_idx: start_idx + 4]

        if len(values) == 4:
            d1, d2, d3, d4 = values
            d1_d3 = d1 - d3
            d2_d4 = d2 - d4

            # 对于0频分量，使用 (D1-D3 + D2-D4)/2 的计算方法
            if fx == 0 and fy == 0:
                dc_component = (d1_d3 + d2_d4) / 2
                print(f"0频分量 (D1-D3 + D2-D4)/2: {dc_component}")
                coeff = dc_component
            else:
                # 其他频点使用复数系数
                coeff = d1_d3 - 1j * d2_d4

            temp_coeffs[(fx, fy)] = coeff

    # 计算归一化因子
    normalization_factor = 1.0
    if dc_component is not None and abs(dc_component) > 1e-9:
        normalization_factor = abs(dc_component)
        print(f"使用0频分量 {dc_component} 进行归一化。")
    else:
        print("警告: 未找到0频分量或其值过小，跳过归一化。")

    # 第二次遍历：生成Excel数据并计算归一化后的值
    for i, (fx, fy) in enumerate(sampled_freq_points):
        start_idx = i * 4
        values = averages[start_idx: start_idx + 4]

        if len(values) == 4:
            d1, d2, d3, d4 = values
            d1_d3 = d1 - d3
            d2_d4 = d2 - d4
            coeff = temp_coeffs[(fx, fy)]

            # 计算归一化后的值
            if normalization_factor > 1e-9:
                normalized_coeff = coeff / normalization_factor
            else:
                normalized_coeff = coeff

            # 存储系数
            fourier_coeffs[(fx, fy)] = coeff
            normalized_coeffs[(fx, fy)] = normalized_coeff

            # 生成Excel行数据，在傅里叶系数和归一化后之间空两列
            if fx == 0 and fy == 0:
                # 特殊处理0频的Excel输出
                row = [f"{fx}, {fy}", d1, d2, d3, d4, "", d1_d3, "", d2_d4, "", f"{coeff:.5f}", "", "", f"{normalized_coeff:.5f}"]
            else:
                # 正常处理其他频点的Excel输出
                row = [f"{fx}, {fy}", d1, d2, d3, d4, "", d1_d3, "", d2_d4, "", f"{coeff.real:.5f} + {coeff.imag:.5f}j", "", "", f"{normalized_coeff.real:.5f} + {normalized_coeff.imag:.5f}j"]

            output_data.append(row)

    # 创建并保存Excel，调整列顺序，在傅里叶系数和归一化后之间空两列
    columns = ["fx/fy", "D1 (0°)", "D2 (90°)", "D3 (180°)", "D4 (270°)", "", "D1-D3", "", "D2-D4", "", "傅里叶系数", "", "", "归一化后"]
    result_df = pd.DataFrame(output_data, columns=columns)
    result_df.to_excel(output_excel_file, index=False)
    print(f"详细计算结果已保存到: {output_excel_file}")
    print("-" * 30)

    # --- 步骤 4: 重建图像 ---
    print("--- 步骤 4: 从稀疏傅里叶系数重建图像 ---")

    # 创建一个空的傅里叶空间矩阵 (k-space)
    k_space = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=complex)

    # 填充测得的傅里叶系数 (包括共轭对称部分)
    for (fx, fy), coeff in fourier_coeffs.items():
        # 计算k空间索引（0频在中心位置）
        # 对于128x128图像，中心位置是(64,64)
        kx = (fx + 64) % IMAGE_SIZE  # fx范围[-64,63]，映射到[0,127]
        ky = (fy + 64) % IMAGE_SIZE  # fy范围[0,63]，映射到[0,63]然后移到中心
        
        # 填充 F(fx, fy)
        if 0 <= kx < IMAGE_SIZE and 0 <= ky < IMAGE_SIZE:
            k_space[ky, kx] = coeff
        
        # 填充共轭对称点 F(-fx, -fy) = conj(F(fx, fy))
        # 排除直流分量
        if fx != 0 or fy != 0:
            kx_conj = (-fx + 64) % IMAGE_SIZE
            ky_conj = (-fy + 64) % IMAGE_SIZE
            if 0 <= kx_conj < IMAGE_SIZE and 0 <= ky_conj < IMAGE_SIZE:
                k_space[ky_conj, kx_conj] = np.conj(coeff)

    # 进行逆傅里叶变换
    # 由于0频在中心，需要使用ifftshift将其移到左上角
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(k_space))
    reconstructed_image_real = np.real(reconstructed_image)

    # 归一化到 0-255
    min_val, max_val = np.min(reconstructed_image_real), np.max(reconstructed_image_real)
    print(f"重构图像亮度范围: 最小值={min_val:.2f}, 最大值={max_val:.2f}")
    
    if max_val > min_val:
        reconstructed_image_norm = 255 * (reconstructed_image_real - min_val) / (max_val - min_val)
    else:
        reconstructed_image_norm = np.zeros_like(reconstructed_image_real)

    reconstructed_image_uint8 = reconstructed_image_norm.astype(np.uint8)

    # 左右翻转图像（只保留翻转后的图像）
    reconstructed_image_flipped = np.fliplr(reconstructed_image_uint8)
    print("已对重构图像进行左右翻转")

    # 保存翻转后的图像
    Image.fromarray(reconstructed_image_flipped).save(output_image_path)
    print(f"翻转后图像已保存到: {output_image_path}")
    print("-" * 30)

    # --- 步骤 5: 性能评估与对比图生成 ---
    print("--- 步骤 5: 图像质量评估与对比图生成 ---")
    try:
        original_img = Image.open(original_image_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        original_array = np.array(original_img)

        # 计算MSE, PSNR, SSIM（基于翻转后的图像）
        mse = np.mean((original_array - reconstructed_image_flipped) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        ssim_val = ssim(original_array, reconstructed_image_flipped, data_range=255)

        print(f"MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}")

        # 创建对比图
        comparison_img = Image.new('RGB', (IMAGE_SIZE * 2 + 60, IMAGE_SIZE + 80), (255, 255, 255))
        draw = ImageDraw.Draw(comparison_img)

        # 粘贴图像
        comparison_img.paste(original_img, (20, 40))
        comparison_img.paste(Image.fromarray(reconstructed_image_flipped), (IMAGE_SIZE + 40, 40))

        # 添加文字
        try:
            font = ImageFont.truetype("msyh.ttc", 16)  # 微软雅黑
        except IOError:
            font = ImageFont.load_default()

        draw.text((20, 10), "原始图像", fill="black", font=font)
        draw.text((IMAGE_SIZE + 40, 10), f"CS重构 ({data_number})", fill="black", font=font)
        info_text = f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}"
        draw.text((2, IMAGE_SIZE + 50), info_text, fill="black", font=font)

        comparison_img.save(output_comparison_path)
        print(f"对比图像已保存到: {output_comparison_path}")

    except Exception as e:
        print(f"生成对比图时发生错误: {e}")

    print("\n所有处理完成！")
    
    # 分析图像偏灰的原因
    print("\n=== 图像偏灰分析 ===")
    print(f"重构图像亮度范围: 最小值={min_val:.2f}, 最大值={max_val:.2f}")
    print(f"归一化后范围: 0-255")
    print("可能的偏灰原因:")
    print("1. 压缩感知采样率较低(40%)，丢失了高频细节")
    print("2. 傅里叶系数填充不完整，特别是高频部分")
    print("3. 数据预处理过程中的噪声影响")
    print("4. 重建算法的局限性")
    print("5. 可能存在相位测量误差")


if __name__ == '__main__':
    main()