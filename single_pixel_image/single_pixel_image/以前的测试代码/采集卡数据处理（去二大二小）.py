"""
去除采集卡采集到的两个最大值和两个最小值，给剩下的值取平均当作测到的电压
修改输入的图像位置的代码在第21行，修改图像的输出位置的代码在178行
"""
import pandas as pd
import numpy as np

# 尝试读取TDMS文件
try:
    # 安装nptdms库（如果未安装）
    try:
        from nptdms import TdmsFile
    except ImportError:
        print("正在安装nptdms库...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nptdms"])
        from nptdms import TdmsFile
    
    # 读取TDMS文件
    input_file = r"D:\桌面\毕设\Data[4,4]\3 倒置.xlsx"
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
                data = channel[:].tolist()
                print(f"成功读取TDMS文件，数据长度: {len(data)}")
                print(f"数据类型: {type(data[0]) if data else '空'}")
                print(f"前5个数据: {data[:5] if data else '无'}")
            else:
                raise Exception("TDMS文件中没有通道")
        else:
            raise Exception("TDMS文件中没有组")
except Exception as e:
    print(f"读取TDMS文件失败: {e}")
    # 尝试读取文本文件
    try:
        print("尝试作为文本文件读取...")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 提取数值数据
        data = []
        for line in lines:
            line = line.strip()
            if line and line.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                try:
                    data.append(float(line))
                except:
                    pass
        print(f"成功读取文本文件，数据长度: {len(data)}")
        print(f"前5个数据: {data[:5] if data else '无'}")
    except Exception as e2:
        print(f"读取文本文件失败: {e2}")
        # 尝试读取CSV文件
        try:
            print("尝试作为CSV文件读取...")
            df = pd.read_csv(input_file)
            data = df.iloc[:, 0].tolist()
            print(f"成功读取CSV文件，数据长度: {len(data)}")
            print(f"前5个数据: {data[:5] if data else '无'}")
        except Exception as e3:
            print(f"读取CSV文件失败: {e3}")
            raise Exception("无法读取文件，请检查文件格式")

# 存储计算结果
averages = []

# 从第二个数据开始，每10个数据为一组
for i in range(1, len(data), 10):  # 从索引1开始（第二个数据）
    # 确保取完整的10个数据
    if i + 10 <= len(data):
        # 提取当前10个数据
        current_group = data[i:i+10]
        # 排序
        sorted_group = sorted(current_group)
        # 去掉前4个最小值
        filtered_group = sorted_group[4:]
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
output_file = r"D:\桌面\毕设\data[16,16]\数据3的处理.xlsx"
result_df.to_excel(output_file, index=False)

print(f"处理完成！结果已保存到 {output_file}")
print("计算得到的平均值：")
print(result_df)