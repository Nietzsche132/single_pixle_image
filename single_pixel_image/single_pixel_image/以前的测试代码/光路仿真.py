from PIL import Image
import numpy as np
import pandas as pd
import os

# 图像路径
input_path = r"D:\桌面\毕设\picture\测试图案\number1_128x128.bmp"

# 输出Excel路径
excel_output = r"D:\桌面\毕设\picture\测试图案\反射率逆推的傅里叶系数.xlsx"

try:
    # 1. 读取图像
    img = Image.open(input_path)
    print(f"成功读取图像: {input_path}")
    print(f"图像尺寸: {img.size}")
    
    # 2. 转换为灰度图像
    if img.mode != 'L':
        img = img.convert('L')
        print("已转换为灰度图像")
    
    # 3. 转换为numpy数组
    img_array = np.array(img)
    print(f"图像数组形状: {img_array.shape}")
    
    # 4. 归一化到[0,1]范围（模拟反射率）
    normalized_img = img_array / 255.0
    print("已归一化图像数据")
    
    # 5. 计算傅里叶变换
    fft_result = np.fft.fft2(normalized_img)
    
    # 6. 准备数据
    m, n = fft_result.shape
    data = []
    
    # 生成频率坐标
    fy_range = np.arange(0, m)
    fx_range = np.arange(0, n)
    
    # 先循环fx，再循环fy，与用户提供的顺序一致
    for j in range(n):  # fx循环
        for i in range(m):  # fy循环
            fx = fx_range[j]
            fy = fy_range[i]
            coeff = fft_result[i, j]
            # 格式化为字符串
            coeff_str = f"{coeff.real:.4f} {'+' if coeff.imag >= 0 else '-'} {abs(coeff.imag):.4f}j"
            data.append([f"{fx}, {fy}", coeff_str])
    
    # 7. 创建DataFrame
    df = pd.DataFrame(data, columns=['fx/fy', '傅里叶系数'])
    
    # 8. 保存到Excel
    df.to_excel(excel_output, index=False)
    
    print(f"\n转换完成！")
    print(f"傅里叶系数已保存到: {excel_output}")
    print(f"总共计算了 {len(data)} 个傅里叶系数")
    
except FileNotFoundError:
    print(f"错误：找不到文件 {input_path}")
except Exception as e:
    print(f"发生错误: {e}")
