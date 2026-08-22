import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from pathlib import Path
from tqdm import tqdm  # 导入进度条库
import itertools  # 用于生成参数组合

# --- 1. 设置文件路径和参数搜索空间 ---

# 文件路径
original_image_path = Path(r"D:\桌面\毕设\picture\测试图案\number1_128x128.png")
reconstructed_image_path = Path(r"D:\桌面\毕设\data（压缩感知）\40%\数据4处理重构.png")
output_dir = Path(r"D:\桌面\毕设\data（压缩感知）\40%")
output_dir.mkdir(exist_ok=True)

# 定义每种滤波器的参数搜索范围 (Parameter Search Space)
# 您可以根据需要调整这些范围
PARAM_SPACE = {
    'mean': {
        'ksize': [3, 5, 7]  # 均值滤波的核大小
    },
    'gaussian': {
        'ksize': [3, 5, 7]  # 高斯滤波的核大小
    },
    'median': {
        'ksize': [3, 5, 7]  # 中值滤波的核大小 (必须是奇数)
    },
    'bilateral': {
        'd': [5, 9],  # 邻域直径
        'sigmaColor': [25, 50, 75],  # 灰度相似性标准差
        'sigmaSpace': [25, 50, 75]  # 空间距离标准差
    }
}

# --- Matplotlib 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 2. 定义辅助函数 (与之前相同，但 create_comparison_plot 稍作修改) ---

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


# --- 3. 核心优化与处理函数 ---

def find_best_params_and_apply(filter_type, original_img, recon_img):
    """
    根据指定的滤波器类型，搜索最优参数并返回结果。

    Args:
        filter_type (str): 'mean', 'gaussian', 'median', or 'bilateral'
        original_img (np.array): 原始参考图像
        recon_img (np.array): 待处理的重构图像

    Returns:
        tuple: (最佳滤波图像, 最佳参数字典, (最佳mse, 最佳psnr, 最佳ssim))
    """
    print(f"\n--- 正在为【{filter_type}】滤波器搜索最优参数... ---")

    best_ssim = -1
    best_params = {}
    best_filtered_image = None

    # 获取参数空间并生成所有参数组合
    params_to_search = PARAM_SPACE[filter_type]
    keys, values = zip(*params_to_search.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 使用 tqdm 显示进度条
    for params in tqdm(param_combinations, desc=f"搜索 {filter_type}"):
        # 应用滤波器
        if filter_type == 'mean':
            filtered_img = cv2.blur(recon_img, (params['ksize'], params['ksize']))
        elif filter_type == 'gaussian':
            filtered_img = cv2.GaussianBlur(recon_img, (params['ksize'], params['ksize']), 0)
        elif filter_type == 'median':
            filtered_img = cv2.medianBlur(recon_img, params['ksize'])
        elif filter_type == 'bilateral':
            filtered_img = cv2.bilateralFilter(recon_img, params['d'], params['sigmaColor'], params['sigmaSpace'])

        # 计算SSIM
        _, _, ssim_val = calculate_metrics(original_img, filtered_img)

        # 如果找到更好的SSIM，则更新记录
        if ssim_val > best_ssim:
            best_ssim = ssim_val
            best_params = params
            best_filtered_image = filtered_img

    # 使用找到的最佳参数计算所有指标
    best_metrics = calculate_metrics(original_img, best_filtered_image)

    # 打印最佳结果
    print(f"【{filter_type}】最佳参数: {best_params}")
    print(f"【{filter_type}】最佳SSIM: {best_metrics[2]:.4f}")

    return best_filtered_image, best_params, best_metrics


# --- 4. 主程序 ---

if __name__ == "__main__":
    try:
        # 加载图像
        original_img = np.array(Image.open(original_image_path).convert('L'))
        reconstructed_img = np.array(Image.open(reconstructed_image_path).convert('L'))
        print("图像加载成功！\n")

        # 对比原始重构图的指标
        print("--- 原始重构图指标 ---")
        initial_metrics = calculate_metrics(original_img, reconstructed_img)
        print(f"MSE: {initial_metrics[0]:.2f} | PSNR: {initial_metrics[1]:.2f} dB | SSIM: {initial_metrics[2]:.4f}")

        # 定义要优化的滤波器类型
        filter_types = ['mean', 'gaussian', 'median', 'bilateral']

        for f_type in filter_types:
            # 寻找最优参数并应用滤波
            best_image, best_params, best_metrics = find_best_params_and_apply(
                f_type, original_img, reconstructed_img
            )

            # 创建标题，包含滤波器类型和找到的最优参数
            filter_name_map = {'mean': '均值', 'gaussian': '高斯', 'median': '中值', 'bilateral': '双边'}
            params_str = ', '.join([f'{k}={v}' for k, v in best_params.items()])
            plot_title = f"CS重构 ({filter_name_map[f_type]}滤波: {params_str})"

            # 生成并保存对比图
            output_filename = output_dir / f"数据4_{filter_name_map[f_type]}滤波_优化后.png"
            create_comparison_plot(
                original_img,
                best_image,
                plot_title,
                best_metrics,
                output_filename
            )

        print("\n🎉 所有滤波器的参数优化和评估已全部完成！")

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件！请检查路径是否正确。\n{e}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")