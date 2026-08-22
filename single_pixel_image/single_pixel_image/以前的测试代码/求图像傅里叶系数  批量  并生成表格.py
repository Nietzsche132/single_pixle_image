##对输入的图像批量求傅傅里叶系数，并且将傅里叶系数存到excel中
##修改图像在100行，修改excel的保存路径在56行
import numpy as np
import pandas as pd
import os
from PIL import Image


def calculate_fourier_coefficients(image_path):
    """
    计算图像在fx和fy范围[0,4]内的所有组合的傅里叶系数并导出为Excel

    参数:
        image_path: 图像文件路径
    """
    # 1. 读取图像并转换为灰度图
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)

    # 2. 计算傅里叶变换
    f_transform = np.fft.fft2(img_array)
    f_transform_shifted = np.fft.fftshift(f_transform)  # 零频移到中心

    # 3. 获取图像中心位置
    rows, cols = img_array.shape
    crow, ccol = rows // 2, cols // 2

    # 4. 生成所有频率组合 (fx从0到4, fy从0到4)
    fx_values = range(0, 5)  # 0,1,2,3,4
    fy_values = range(0, 5)  # 0,1,2,3,4

    # 5. 创建结果数据
    data = []

    for fx in fx_values:
        for fy in fy_values:
            # 获取指定频率的系数
            coeff = f_transform_shifted[crow + fy, ccol + fx]
            magnitude = np.abs(coeff)
            phase = np.angle(coeff)

            # 将复数转换为a+bj格式的字符串
            complex_str = f"{coeff.real:.6f}+{coeff.imag:.6f}j"

            data.append({
                '频率点': f'fx={fx},fy={fy}',
                '复数': complex_str,
                '幅度': magnitude,
                '相位': phase
            })

    # 6. 创建DataFrame
    df = pd.DataFrame(data)

    # 7. 定义保存路径
    save_dir = r"D:\桌面\毕设\picture\傅里叶基底1bit（1024×1024）"

    # 如果目录不存在，创建目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建目录: {save_dir}")

    # 8. 从图像路径中提取文件名
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    excel_filename = f"{base_name}_fourier_coeff.xlsx"
    excel_path = os.path.join(save_dir, excel_filename)

    # 9. 导出到Excel，按照您要求的格式
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 写入主表格
        df[['频率点', '复数', '幅度', '相位']].to_excel(writer, sheet_name='傅里叶系数', index=False)

        # 获取工作簿和工作表进行格式调整
        workbook = writer.book
        worksheet = writer.sheets['傅里叶系数']

        # 设置列宽
        worksheet.column_dimensions['A'].width = 15  # 频率点列
        worksheet.column_dimensions['B'].width = 25  # 复数列
        worksheet.column_dimensions['C'].width = 15  # 幅度列
        worksheet.column_dimensions['D'].width = 15  # 相位列

        # 设置数字格式
        for row in range(2, len(df) + 2):  # Excel行号从1开始，第1行是表头
            # 幅度列设置为科学计数法格式
            worksheet.cell(row=row, column=3).number_format = '0.0000E+00'
            # 相位列设置为保留6位小数
            worksheet.cell(row=row, column=4).number_format = '0.000000'

    print(f"傅里叶系数已保存到: {excel_path}")
    print(f"表格格式: 共{len(df)}行，包含fx和fy从0到4的所有组合")

    return df, excel_path


# 使用示例
if __name__ == "__main__":
    # 设置图像路径
    image_path = r"D:\桌面\毕设\picture\测试图案\number1_128x128.bmp"

    try:
        # 计算傅里叶系数并导出Excel
        df_results, saved_path = calculate_fourier_coefficients(image_path)

        # 打印前几行结果
        print("\n表格前几行预览:")
        print(df_results.head(10))

        print(f"\nExcel文件保存在: {saved_path}")
        print("\n表格包含以下列:")
        print("1. 频率点: fx和fy的组合，格式为'fx=?,fy=?'")
        print("2. 复数: 复数形式的傅里叶系数，格式为'a+bj'")
        print("3. 幅度: 傅里叶系数的幅度值")
        print("4. 相位: 傅里叶系数的相位值（弧度）")

    except FileNotFoundError:
        print(f"错误: 找不到图像文件 {image_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")