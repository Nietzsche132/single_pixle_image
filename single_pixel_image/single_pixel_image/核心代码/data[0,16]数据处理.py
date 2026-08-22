"""
采集卡数据处理整合版
功能：从第二个数据开始，每十个数据为一组，去除2个最大值和2个最小值，然后取平均
并生成重构图像
"""
import pandas as pd
import numpy as np
from PIL import Image
# ==================== 参数配置 ====================
# 数据编号（只需修改这里）
data_number = 17
# 数据目录
data_dir = r"D:\桌面\毕设\data[0,16]"
# 输入文件路径
input_file = f"{data_dir}\{data_number}.tdms"
# 输出Excel文件路径
output_file = f"{data_dir}\数据{data_number}处理结果.xlsx"
# 输出重构图像路径
output_image_path = f"{data_dir}\数据{data_number}处理重构.png"
# 输出傅里叶系数幅度图路径
output_fft_path = f"{data_dir}\数据{data_number}处理傅里叶系数幅度.png"
# 输出对比图像路径
output_comparison_path = f"{data_dir}\数据{data_number}处理对比.png"
# ==================================================

# 原始图像路径
original_image_path = r"D:\桌面\毕设\picture\测试图案\number1_128x128.png"
# 初始化数据变量
data = []
# 尝试读取TDMS文件
try:
    from nptdms import TdmsFile
    
    # 读取TDMS文件
    print(f"正在读取TDMS文件: {input_file}")
    
    with TdmsFile.open(input_file) as tdms_file:
        # 打印TDMS文件结构
        print("TDMS文件结构:")
        for group in tdms_file.groups():
            print(f"  组: {group.name}")
            for channel in group.channels():
                print(f"    通道: {channel.name}")
        
        # 尝试获取第一个通道的数据
        if tdms_file.groups():
            group = tdms_file.groups()[0]
            if group.channels():
                channel = group.channels()[0]
                # 确保数据可以转换为列表
                channel_data = channel[:]
                data = channel_data.tolist() if hasattr(channel_data, 'tolist') else list(channel_data)
                print(f"成功读取TDMS文件，数据长度: {len(data)}")
                print(f"数据类型: {type(data[0]) if data else '空'}")
                print(f"前5个数据: {data[:5] if data else '无'}")
            else:
                raise Exception("TDMS文件中没有通道")
        else:
            raise Exception("TDMS文件中没有组")
except ImportError:
    print("错误：缺少nptdms库，请先安装: pip install nptdms")
    raise
except Exception as e:
    print(f"读取TDMS文件失败: {e}")
    # 尝试读取文本文件
    try:
        print("尝试作为文本文件读取...")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 提取数值数据
        for line in lines:
            line = line.strip()
            if line and line.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                try:
                    data.append(float(line))
                except ValueError:
                    pass
        print(f"成功读取文本文件，数据长度: {len(data)}")
        print(f"前5个数据: {data[:5] if data else '无'}")
    except Exception as e2:
        print(f"读取文本文件失败: {e2}")
        # 尝试读取CSV文件
        try:
            print("尝试作为CSV文件读取...")
            df = pd.read_csv(input_file)
            # 确保获取的是数值列
            if not df.empty:
                data = df.iloc[:, 0].tolist()
            print(f"成功读取CSV文件，数据长度: {len(data)}")
            print(f"前5个数据: {data[:5] if data else '无'}")
        except Exception as e3:
            print(f"读取CSV文件失败: {e3}")
            raise Exception("无法读取文件，请检查文件格式")

# 存储计算结果
averages = []

# 从第二个数据开始，每10个数据为一组
if data:
    for i in range(1, len(data), 10):  # 从索引1开始（第二个数据）
        # 确保取完整的10个数据
        if i + 10 <= len(data):
            # 提取当前10个数据
            current_group = data[i:i+10]
            # 排序
            sorted_group = sorted(current_group)
            # 去除2个最大值和2个最小值
            filtered_group = sorted_group[2:-2]
            # 计算平均值
            if filtered_group:  # 确保有数据
                avg = sum(filtered_group) / len(filtered_group)
                averages.append(avg)

print(f"计算完成，得到 {len(averages)} 个平均值")

# 准备输出数据
output_data = []

# 定义fx和fy的组合（0到16）
fx_fy_pairs = []
# 生成fx从0到16，fy从0到16的所有组合
for fx in range(17):  # fx=0,1,2,...,16
    for fy in range(17):  # fy=0,1,2,...,16
        fx_fy_pairs.append((fx, fy))

print(f"生成 {len(fx_fy_pairs)} 个fx/fy组合")

# 处理每个fx, fy组合
for i, (fx, fy) in enumerate(fx_fy_pairs):
    if fx == 0 and fy == 0:
        # 特殊处理：只有fx和fy都为0时
        if len(averages) > 0:
            output_data.append([f"{fx}, {fy}", "", "", averages[0], ""])
    else:
        # 其他情况：每个组合对应4个值
        start_idx = 1 + (i - 1) * 4  # 从第二个值开始，每个组合4个值
        end_idx = start_idx + 4
        values = averages[start_idx:end_idx] if start_idx < len(averages) else []
        # 填充不足4个值的情况
        while len(values) < 4:
            values.append("")
        output_data.append([f"{fx}, {fy}", *values])

# 创建结果DataFrame，添加空列作为间隔
columns = ["fx/fy", "D1 (0°)", "D2 (90°)", "D3 (180°)", "D4 (270°)", "", "D1-D3", "", "D2-D4", "", "傅里叶系数"]
# 扩展每行数据，添加空列
for i in range(len(output_data)):
    # 在适当位置插入空字符串作为间隔
    row = output_data[i]
    # 原始数据: [fx/fy, D1, D2, D3, D4]
    # 新格式: [fx/fy, D1, D2, D3, D4, "", D1-D3, "", D2-D4, "", 傅里叶系数]
    # 先扩展到11列
    while len(row) < 11:
        row.append("")
    output_data[i] = row

result_df = pd.DataFrame(output_data, columns=columns)

# 确保D1-D3和D2-D4列是数值类型
result_df['D1-D3'] = np.nan
result_df['D2-D4'] = np.nan

# 从fx=0, fy=1开始计算（索引从1开始）
for i in range(1, len(result_df)):
    row = result_df.iloc[i]
    
    # 计算D1-D3
    d1 = row['D1 (0°)']
    d3 = row['D3 (180°)']
    if pd.notna(d1) and pd.notna(d3) and not isinstance(d1, str) and not isinstance(d3, str):
        d1_d3 = float(d1) - float(d3)
        result_df.loc[i, 'D1-D3'] = d1_d3
    
    # 计算D2-D4
    d2 = row['D2 (90°)']
    d4 = row['D4 (270°)']
    if pd.notna(d2) and pd.notna(d4) and not isinstance(d2, str) and not isinstance(d4, str):
        d2_d4 = float(d2) - float(d4)
        result_df.loc[i, 'D2-D4'] = d2_d4
    
    # 计算傅里叶系数
    d1_d3_val = result_df.loc[i, 'D1-D3']
    d2_d4_val = result_df.loc[i, 'D2-D4']
    if pd.notna(d1_d3_val) and pd.notna(d2_d4_val):
        # 这里存储为字符串表示复数
        d1_d3_val = float(d1_d3_val)
        d2_d4_val = float(d2_d4_val)
        # 傅里叶系数 = D1-D3 - j*(D2-D4)
        # 处理符号，将"- -"替换为"+"
        if d2_d4_val < 0:
            # 如果d2_d4为负，那么 -j*(-value) = +j*value
            fourier = f"{d1_d3_val:.5f} + {abs(d2_d4_val):.5f}j"
        else:
            fourier = f"{d1_d3_val:.5f} - {d2_d4_val:.5f}j"
        result_df.loc[i, '傅里叶系数'] = fourier

# 保存到新的Excel文件
result_df.to_excel(output_file, index=False)

print(f"处理完成！结果已保存到 {output_file}")
print("计算得到的平均值：")
print(result_df)

# ==================== 生成重构图像 ====================
print("\n开始生成重构图像...")

# 创建17x17的傅里叶系数矩阵（fx和fy从0到16）
fourier_matrix_17x17 = np.zeros((17, 17), dtype=complex)

# 填充傅里叶系数矩阵
for i, row in result_df.iterrows():
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
            real_part = 0.0
            imag_part = 0.0
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
                real_part = float(fourier_str)
                fourier_matrix_17x17[fy, fx] = complex(real_part, 0)

# 处理(0,0)位置的特殊情况
# 从D3列获取(0,0)的值（直流分量）
if not result_df.empty and pd.notna(result_df.iloc[0]['D3 (180°)']) and not isinstance(result_df.iloc[0]['D3 (180°)'], str):
    # (0,0)位置交换后仍然是(0,0)
    fourier_matrix_17x17[0, 0] = complex(float(result_df.iloc[0]['D3 (180°)']), 0)

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
if max_val > min_val:
    reconstructed_image_normalized = ((reconstructed_image_real - min_val) / (max_val - min_val)) * 255
else:
    reconstructed_image_normalized = np.zeros_like(reconstructed_image_real)
reconstructed_image_normalized = reconstructed_image_normalized.astype(np.uint8)

print("\n重构的128x128图像数据范围：")
print(f"最小值: {np.min(reconstructed_image_normalized)}")
print(f"最大值: {np.max(reconstructed_image_normalized)}")
print(f"平均值: {np.mean(reconstructed_image_normalized)}")

# 重构完成，不再单独保存重构图像，只生成对比图像

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

fft_img = Image.fromarray(amplified_fft, mode='L')
fft_img.save(output_fft_path)

print(f"傅里叶系数幅度图已保存到 {output_fft_path}")

# ==================== 计算MSE和PSNR并生成对比图像 ====================
print("\n开始计算MSE和PSNR...")

try:
    # 读取原始图像
    original_img = Image.open(original_image_path).convert('L')
    original_img = original_img.resize((128, 128))  # 确保尺寸一致
    original_array = np.array(original_img)
    
    print(f"成功读取原始图像: {original_image_path}")
    print(f"原始图像尺寸: {original_img.size}")
    
    # 计算MSE
    mse = np.mean((original_array - reconstructed_image_normalized) ** 2)
    
    # 计算PSNR
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else float('inf')
    
    print(f"\n计算结果：")
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr_value:.2f} dB")
    
    # 生成对比图像（原始图像 + 重构图像）- 保持原始尺寸，提高文字清晰度
    from PIL import ImageDraw, ImageFont
    
    # 设置尺寸和边距
    img_size = 128
    margin = 30
    title_height = 40
    info_height = 45
    gap = 20  # 两张图像之间的间距
    
    comparison_width = img_size * 2 + gap + margin * 2
    comparison_height = img_size + title_height + info_height + margin
    comparison_img = Image.new('RGB', (comparison_width, comparison_height), color=(255, 255, 255))
    
    # 直接使用原始尺寸的图像
    original_img = Image.fromarray(original_array, mode='L')
    reconstructed_img = Image.fromarray(reconstructed_image_normalized, mode='L')
    
    # 绘制原始图像
    comparison_img.paste(original_img, (margin, title_height))
    
    # 绘制重构图像
    comparison_img.paste(reconstructed_img, (margin + img_size + gap, title_height))
    
    # 添加标题和数据
    draw = ImageDraw.Draw(comparison_img)
    
    # 尝试使用系统中可用的中文字体 - 使用更大字号
    title_font = None
    info_font = None
    try:
        # 尝试使用宋体
        title_font = ImageFont.truetype('simsun.ttc', 20)
        info_font = ImageFont.truetype('simsun.ttc', 16)
    except (IOError, OSError):
        try:
            # 尝试使用黑体
            title_font = ImageFont.truetype('simhei.ttf', 20)
            info_font = ImageFont.truetype('simhei.ttf', 16)
        except (IOError, OSError):
            # 尝试使用微软雅黑
            try:
                title_font = ImageFont.truetype('msyh.ttc', 20)
                info_font = ImageFont.truetype('msyh.ttc', 16)
            except (IOError, OSError):
                # 使用默认字体
                title_font = ImageFont.load_default()
                info_font = ImageFont.load_default()
    
    # 添加标题（在图像上方居中）
    draw.text((margin + img_size // 2, 15), "原始图像", fill=(0, 0, 0), font=title_font, anchor="mm")
    draw.text((margin + img_size + gap + img_size // 2, 15), "重构图像", fill=(0, 0, 0), font=title_font, anchor="mm")
    
    # 添加MSE和PSNR（在右下角，带背景框）
    mse_psnr_text = f"MSE: {mse:.2f}, PSNR: {psnr_value:.2f} dB"
    
    # 计算文本尺寸
    bbox = draw.textbbox((0, 0), mse_psnr_text, font=info_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 背景框位置和尺寸
    box_padding = 10
    box_x = comparison_width - text_width - box_padding * 2 - margin // 2
    box_y = title_height + img_size + 10
    box_width = text_width + box_padding * 2
    box_height = text_height + box_padding * 2
    
    # 绘制黑色边框背景
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], outline=(0, 0, 0), width=2)
    
    # 绘制白色填充
    draw.rectangle([box_x + 2, box_y + 2, box_x + box_width - 2, box_y + box_height - 2], fill=(255, 255, 255))
    
    # 绘制文本（在框内居中）
    text_x = box_x + box_padding
    text_y = box_y + box_padding - 2
    draw.text((text_x, text_y), mse_psnr_text, fill=(0, 0, 0), font=info_font)
    
    # 保存对比图像（使用高质量设置）
    comparison_img.save(output_comparison_path, quality=95)
    print(f"对比图像已保存到 {output_comparison_path}")
    
except Exception as e:
    print(f"计算MSE和PSNR时出错: {e}")
    print("请确保原始图像路径正确且图像格式正确")

print("\n所有处理完成！")
