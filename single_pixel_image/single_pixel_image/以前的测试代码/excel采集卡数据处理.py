"""
去除采集卡采集到的两个最大值和两个最小值，给剩下的值取平均当作测到的电压
修改输入的Excel文件位置在代码第8行，修改输出位置在代码第142行
"""
import pandas as pd
import numpy as np

input_file = r"D:\桌面\毕设\Data[4,4]\处理2.xlsx"
print(f"正在读取Excel文件: {input_file}")

# 读取Excel文件的第二个工作表（数据工作表）
print(f"正在读取Excel文件: {input_file}")

# 检查工作表
xls = pd.ExcelFile(input_file)
print("工作表:", xls.sheet_names)

# 读取合适的工作表（如果有多个工作表，读取第二个，否则读取第一个）
sheet_index = 1 if len(xls.sheet_names) > 1 else 0
df = pd.read_excel(input_file, sheet_name=sheet_index)
print(f"读取工作表: {xls.sheet_names[sheet_index]}")

# 清理数据，确保只保留数值
print("Excel文件信息：")
print(f"列数: {len(df.columns)}")
print(f"行数: {len(df)}")
print("列名:", df.columns.tolist())

print("\n原始数据前10行预览：")
print(df.head(10))

# 查找包含数值数据的列
data = []
for col in df.columns:
    col_data = []
    for val in df[col]:
        try:
            num = float(val)
            if not np.isnan(num):  # 跳过NaN值
                col_data.append(num)
        except (ValueError, TypeError):
            pass
    if len(col_data) > 10:  # 找到有足够数据的列
        data = col_data
        print(f"\n找到数据列: {col}")
        print(f"数据长度: {len(data)}")
        print(f"前5个数据: {data[:5]}")
        print(f"最后5个数据: {data[-5:]}")
        break

if not data:
    # 如果没有找到，尝试读取所有数值
    print("\n未找到数据列，尝试读取所有数值...")
    for col in df.columns:
        for val in df[col]:
            try:
                num = float(val)
                if not np.isnan(num):
                    data.append(num)
            except (ValueError, TypeError):
                pass
    print(f"总数据长度: {len(data)}")

print(f"\n成功读取Excel文件，有效数据长度: {len(data)}")
print(f"数据类型: {type(data[0]) if data else '空'}")
print(f"前5个数据: {data[:5] if data else '无'}")
print(f"最后5个数据: {data[-5:] if data else '无'}")

averages = []

for i in range(0, len(data), 10):
    if i + 10 <= len(data):
        current_group = data[i:i+10]
        sorted_group = sorted(current_group)
        filtered_group = sorted_group[2:-2]
        if filtered_group:
            avg = sum(filtered_group) / len(filtered_group)
            averages.append(avg)

print(f"计算完成，得到 {len(averages)} 个平均值")

output_data = []

fx_fy_pairs = []
for fx in range(5):
    for fy in range(5):
        fx_fy_pairs.append((fx, fy))

print(f"生成 {len(fx_fy_pairs)} 个fx/fy组合")

for i, (fx, fy) in enumerate(fx_fy_pairs):
    if fx == 0 and fy == 0:
        if len(averages) > 0:
            output_data.append([f"{fx}, {fy}", "", "", averages[0], ""])
    else:
        start_idx = 1 + (i - 1) * 4
        end_idx = start_idx + 4
        values = averages[start_idx:end_idx] if start_idx < len(averages) else []
        while len(values) < 4:
            values.append("")
        output_data.append([f"{fx}, {fy}", *values])

columns = ["fx/fy", "D1 (0°)", "D2 (90°)", "D3 (180°)", "D4 (270°)", "", "D1-D3", "", "D2-D4", "", "傅里叶系数"]

for i in range(len(output_data)):
    row = output_data[i]
    while len(row) < 11:
        row.append("")
    output_data[i] = row

result_df = pd.DataFrame(output_data, columns=columns)

result_df['D1-D3'] = np.nan
result_df['D2-D4'] = np.nan

for i in range(1, len(result_df)):
    row = result_df.iloc[i]
    
    d1 = row['D1 (0°)']
    d3 = row['D3 (180°)']
    if pd.notna(d1) and pd.notna(d3) and not isinstance(d1, str) and not isinstance(d3, str):
        d1_d3 = float(d1) - float(d3)
        result_df.loc[i, 'D1-D3'] = d1_d3
    
    d2 = row['D2 (90°)']
    d4 = row['D4 (270°)']
    if pd.notna(d2) and pd.notna(d4) and not isinstance(d2, str) and not isinstance(d4, str):
        d2_d4 = float(d2) - float(d4)
        result_df.loc[i, 'D2-D4'] = d2_d4
    
    d1_d3_val = result_df.loc[i, 'D1-D3']
    d2_d4_val = result_df.loc[i, 'D2-D4']
    if pd.notna(d1_d3_val) and pd.notna(d2_d4_val):
        d1_d3_val = float(d1_d3_val)
        d2_d4_val = float(d2_d4_val)
        if d2_d4_val < 0:
            fourier = f"{d1_d3_val:.5f} + {abs(d2_d4_val):.5f}j"
        else:
            fourier = f"{d1_d3_val:.5f} - {d2_d4_val:.5f}j"
        result_df.loc[i, '傅里叶系数'] = fourier

output_file = r"D:\桌面\毕设\Data[4,4]\数据2的处理.xlsx"
result_df.to_excel(output_file, index=False)

print(f"处理完成！结果已保存到 {output_file}")
print("计算得到的平均值：")
print(result_df)
