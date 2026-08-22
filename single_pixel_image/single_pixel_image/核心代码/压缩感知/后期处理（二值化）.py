import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from pathlib import Path

# --- 1. 设置文件路径 ---

# 请确保这里的路径是正确的
original_image_path = Path(r"D:\桌面\毕设\picture\测试图案\number1_128x128.png")
reconstructed_image_path = Path(r"D:\桌面\毕设\data（压缩感知）\40%\数据4处理重构.png")
output_dir = Path(r"D:\桌面\毕设\data（压缩感知）\40%")

# 确保输出目录存在
output_dir.mkdir(exist_ok=True)

# --- Matplotlib 中文显示设置 ---
# 如果您的系统没有 'SimHei' 字体, 可以换成 'Microsoft YaHei' 等
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置警告: {e}\n图表中的中文可能显示为方块。")


# --- 2. 定义辅助函数 (与之前相同) ---

def calculate_metrics(original_img, processed_img):
    """计算并返回 MSE, PSNR, SSIM 三个指标"""
    mse = mean_squared_error(original_img, processed_img)
    psnr = peak_signal_noise_ratio(original_img, processed_img, data_range=255)
    ssim_val = structural_similarity(original_img, processed_img, data_range=255)
    return mse, psnr, ssim_val


def create_comparison_plot(original_img, processed_img, title, metrics, save_path):
    """生成并保存对比图"""
    mse, psnr, ssim_val = metrics
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('原始图像', fontsize=16)
    axes[0].axis('off')
    axes[1].imshow(processed_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(title, fontsize=16)
    axes[1].axis('off')
    metric_text = f"MSE: {mse:.2f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}"
    fig.text(0.5, 0.15, metric_text, ha='center', va='center', fontsize=14,
             bbox=dict(facecolor='white', alpha=0.5, pad=5))
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"✔️ 对比图已保存至: {save_path}")
    plt.close(fig)


# --- 3. 主程序 ---
if __name__ == "__main__":
    try:
        # 步骤 1: 加载图像
        original_img = np.array(Image.open(original_image_path).convert('L'))
        reconstructed_img = np.array(Image.open(reconstructed_image_path).convert('L'))
        print("图像加载成功！")

        # 步骤 2: 应用二值化处理
        print("\n--- 正在应用二值化处理 (阈值=127) ---")

        # 使用 NumPy 的高效方法进行阈值处理
        # 像素值 > 127 的变为 255，否则变为 0
        thresholded_img = np.where(reconstructed_img > 127, 255, 0).astype(np.uint8)

        # # 这是另一种等效的实现方法 (使用OpenCV)
        # _, thresholded_img_cv = cv2.threshold(reconstructed_img, 127, 255, cv2.THRESH_BINARY)

        print("二值化处理完成！")

        # 步骤 3: 计算处理后图像的各项指标
        print("\n--- 正在计算处理后图像的指标 ---")
        final_metrics = calculate_metrics(original_img, thresholded_img)
        print(
            f"最终指标 - MSE: {final_metrics[0]:.2f} | PSNR: {final_metrics[1]:.2f} dB | SSIM: {final_metrics[2]:.4f}")

        # 步骤 4: 生成并保存最终的对比图
        plot_title = "二值化处理 (阈值=127)"
        output_filename = output_dir / "数据4_二值化处理后.png"

        create_comparison_plot(
            original_img,
            thresholded_img,
            plot_title,
            final_metrics,
            output_filename
        )

        print("\n🎉 全部处理流程已成功完成！")

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件！请检查路径是否正确。\n{e}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")