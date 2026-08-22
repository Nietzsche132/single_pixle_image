"""
绘制TDMS文件数据幅度图表
功能：读取TDMS文件，从第二个数据开始，每10个数据为一组，
去掉两个最大值和两个最小值，对剩下的数值取平均值，
绘制平均值的图像和处理后数据的频谱图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
plt.rc("font",family='YouYuan')

# ==================== 参数配置 ====================
# TDMS文件路径
tdms_file_path = r"D:\桌面\毕设\data[-32,32]\1.tdms"
avg_output_path = r"D:\桌面\毕设\data[-32,32]\数据1平均值图表.png"
spectrum_output_path = r"D:\桌面\毕设\data[-32,32]\数据1处理后频谱图.png"
filtered_output_path = r"D:\桌面\毕设\data[-32,32]\滤去直流分量.xlsx"

# 采样参数
sample_rate = 100  # 处理后的数据采样频率，单位Hz（每10ms一个数据点）
# ==================================================

# 尝试读取TDMS文件
try:
    from nptdms import TdmsFile
    
    print(f"正在读取TDMS文件: {tdms_file_path}")
    
    with TdmsFile.open(tdms_file_path) as tdms_file:
        # 打印TDMS文件结构
        print("TDMS文件结构:")
        for group in tdms_file.groups():
            print(f"  组: {group.name}")
            for channel in group.channels():
                print(f"    通道: {channel.name}")
        
        # 获取第一个通道的数据
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
    raise

# 从第二个数据开始（索引1）
if len(data) >= 2:
    # 从第二个数据开始，每10个数据为一组，计算平均值
    data_values = data[1:]  # 从索引1开始，对应数据个数2
    averages = []
    
    print(f"原始数据长度: {len(data_values)}")
    
    # 数据分组处理
    print("\n正在进行数据分组处理...")
    for i in range(0, len(data_values), 10):
        # 提取当前10个数据
        current_group = data_values[i:i+10]
        if len(current_group) >= 6:  # 确保至少有6个数据
            # 排序
            sorted_group = sorted(current_group)
            # 去掉2个最大值和2个最小值
            filtered_group = sorted_group[2:-2]
            # 计算平均值
            avg = sum(filtered_group) / len(filtered_group)
            averages.append(avg)
    
    print(f"处理后的数据长度: {len(averages)}")
    print(f"平均值个数: {len(averages)}")
    
    if averages:
        # 生成平均值的索引
        avg_indices = list(range(1, len(averages) + 1))
        
        # 绘制平均值图表
        plt.figure(figsize=(12, 6))
        plt.plot(avg_indices, averages, linewidth=0.5)
        plt.title('采集卡采集的一维信号')
        plt.xlabel('信号个数')
        plt.ylabel('幅度（V）')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存平均值图表
        plt.savefig(avg_output_path, dpi=150)
        print(f"平均值图表已保存到: {avg_output_path}")
        
        # 去除直流分量
        print("\n正在去除直流分量...")
        # 计算平均值（直流分量）
        dc_component = np.mean(averages)
        # 减去直流分量
        filtered_averages = [x - dc_component for x in averages]
        print(f"直流分量值: {dc_component}")
        
        # 将滤波后的数据输出到Excel文件
        print("\n正在将滤波后的数据输出到Excel文件...")
        df = pd.DataFrame({
            '索引': avg_indices,
            '原始平均值': averages,
            '滤去直流分量后': filtered_averages
        })
        df.to_excel(filtered_output_path, index=False)
        print(f"滤波后的数据已保存到: {filtered_output_path}")
        
        # 对处理后的数据做傅里叶变换，绘制频谱图
        print("\n正在进行傅里叶变换...")
        n = len(filtered_averages)
        freq = np.fft.fftfreq(n, 1/sample_rate)
        fft_values = np.fft.fft(filtered_averages)
        amplitude = np.abs(fft_values) / n  # 归一化幅度
        
        # 只显示正频率部分
        positive_freq = freq[:n//2]
        positive_amplitude = amplitude[:n//2]
        
        # 绘制频谱图
        plt.figure(figsize=(12, 6))
        plt.plot(positive_freq, positive_amplitude, linewidth=0.5)
        plt.title('处理后数据频谱图')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('幅度')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存频谱图
        plt.savefig(spectrum_output_path, dpi=150)
        print(f"处理后数据频谱图已保存到: {spectrum_output_path}")
        
        # 显示图表
        plt.show()
    else:
        print("没有足够的数据进行处理")
else:
    print("数据长度不足2，无法绘制图表")
