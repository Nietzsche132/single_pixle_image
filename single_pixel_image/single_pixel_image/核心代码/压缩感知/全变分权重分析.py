import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
tv_weights = [0.03, 0.06, 0.10, 0.13, 0.14,0.145, 0.15, 0.16, 0.17, 0.19, 0.20, 0.23, 0.26, 0.30, 0.33, 0.36, 0.40]
mse_values = [95.84, 40.46, 41.96, 40.77, 40.94, 41.23,41.39, 42.14, 42.57, 43.09, 43.80, 45.25, 46.16, 47.36, 48.68, 49.39, 51.18]
psnr_values = [28.32, 32.06, 31.90, 32.03, 32.01, 31.98,31.96, 31.88, 31.84, 31.79, 31.72, 31.58, 31.49, 31.38, 31.26, 31.19, 31.04]
ssim_values = [0.069, 0.256, 0.236, 0.310, 0.313, 0.315,0.314, 0.314, 0.314, 0.314, 0.314, 0.311, 0.308, 0.302, 0.298, 0.294, 0.288]

# 找到权重为0.145的索引
target_weight = 0.145
target_index = tv_weights.index(target_weight)
target_mse = mse_values[target_index]
target_psnr = psnr_values[target_index]
target_ssim = ssim_values[target_index]

# 输出路径
output_dir = r"D:\桌面\毕设\data（压缩感知）\40%\加权全平均法"

# 绘制MSE图表
plt.figure(figsize=(10, 6))
plt.plot(tv_weights, mse_values, 'o-', color='blue', label='MSE')
# 标注目标点
plt.scatter([target_weight], [target_mse], color='red', s=100, zorder=5, facecolors='none', linewidth=2)
plt.text(target_weight + 0.01, target_mse + 1, f'权重={target_weight}\nMSE={target_mse}', 
         fontsize=10, color='red')
plt.title('全变分权重对MSE的影响')
plt.xlabel('全变分权重')
plt.ylabel('MSE')
plt.xlim(0, 0.45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
mse_path = f"{output_dir}\全变分权重对MSE的影响.png"
plt.savefig(mse_path, dpi=300, bbox_inches='tight')
plt.close()

# 绘制PSNR图表
plt.figure(figsize=(10, 6))
plt.plot(tv_weights, psnr_values, 'o-', color='green', label='PSNR')
# 标注目标点
plt.scatter([target_weight], [target_psnr], color='red', s=100, zorder=5, facecolors='none', linewidth=2)
plt.text(target_weight + 0.01, target_psnr + 0.05, f'权重={target_weight}\nPSNR={target_psnr} dB', 
         fontsize=10, color='red')
plt.title('全变分权重对PSNR的影响')
plt.xlabel('全变分权重')
plt.ylabel('PSNR (dB)')
plt.xlim(0, 0.45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
psnr_path = f"{output_dir}\全变分权重对PSNR的影响.png"
plt.savefig(psnr_path, dpi=300, bbox_inches='tight')
plt.close()

# 绘制SSIM图表
plt.figure(figsize=(10, 6))
plt.plot(tv_weights, ssim_values, 'o-', color='red', label='SSIM')
# 标注目标点
plt.scatter([target_weight], [target_ssim], color='blue', s=100, zorder=5, facecolors='none', linewidth=2)
plt.text(target_weight + 0.01, target_ssim + 0.02, f'权重={target_weight}\nSSIM={target_ssim}', 
         fontsize=10, color='blue')
plt.title('全变分权重对SSIM的影响')
plt.xlabel('全变分权重')
plt.ylabel('SSIM')
plt.xlim(0, 0.45)
plt.ylim(0, 0.5)  # SSIM范围0-0.5
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
ssim_path = f"{output_dir}\全变分权重对SSIM的影响.png"
plt.savefig(ssim_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"图表已保存：")
print(f"MSE图表：{mse_path}")
print(f"PSNR图表：{psnr_path}")
print(f"SSIM图表：{ssim_path}")
print(f"\n标注点信息：")
print(f"权重：{target_weight}")
print(f"MSE：{target_mse}")
print(f"PSNR：{target_psnr} dB")
print(f"SSIM：{target_ssim}")

# 绘制MSE和SSIM在同一张图的图表
plt.figure(figsize=(10, 6))

# 左侧Y轴：SSIM
ax1 = plt.gca()
color = 'tab:red'
ax1.set_xlabel('全变分权重')
ax1.set_ylabel('SSIM', color=color)
ax1.plot(tv_weights, ssim_values, 'o-', color=color, label='SSIM')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim(0, 0.45)
ax1.set_ylim(0, 0.5)  # SSIM范围0-0.5
ax1.grid(True, alpha=0.3)

# 右侧Y轴：MSE
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('MSE', color=color)
ax2.plot(tv_weights, mse_values, 's-', color=color, label='MSE')
ax2.tick_params(axis='y', labelcolor=color)

# 在权重为0.145处画竖线
plt.axvline(x=target_weight, color='green', linestyle='--', linewidth=2, label=f'权重={target_weight}')

# 标注0.145时的MSE和SSIM数值
ax1.text(target_weight + 0.02, target_ssim + 0.01, f'SSIM={target_ssim}', 
         fontsize=10, color='red')
ax2.text(target_weight + 0.02, target_mse + 3, f'MSE={target_mse}', 
         fontsize=10, color='blue')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('全变分权重对SSIM和MSE的影响')
plt.tight_layout()

# 保存图表
combined_path = f"{output_dir}\全变分权重对SSIM和MSE的影响.png"
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n组合图表已保存：")
print(f"MSE和SSIM组合图表：{combined_path}")