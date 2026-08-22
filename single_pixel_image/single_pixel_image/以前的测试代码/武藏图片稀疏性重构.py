import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct
from sklearn.linear_model import Lasso
from sklearn.feature_extraction import DictVectorizer
from skimage.metrics import structural_similarity as ssim
import cv2
import warnings
import matplotlib
plt.rc("font",family='YouYuan')

warnings.filterwarnings('ignore')


def compressive_sensing_image_reconstruction(image_path, measurement_rate=0.3):
    """
    基于稀疏性的图像压缩感知重构
    """
    try:
        # 1. 读取图片
        print("正在读取图片...")
        img = Image.open(image_path)
        img_array = np.array(img)

        # 转换为灰度图以简化处理
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
            ##对三维图像的RGB像素值简单地取(R + G + B) / 3当作这个像素灰度图的平均灰度值
        else:
            img_gray = img_array

        # 调整图片大小以提高处理速度
        h, w = img_gray.shape
        if max(h, w) > 256:
            scale_factor = 256 / max(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            img_gray = cv2.resize(img_gray, (new_w, new_h))
            h, w = img_gray.shape
            print(f"图片已调整大小至: {h}x{w}")

        # 2. 分析图片在不同变换域的稀疏性
        print("\n分析图片稀疏性...")

        # DCT域稀疏性分析
        dct_coeff = dct(dct(img_gray.T, norm='ortho').T, norm='ortho')
        ###对灰度图进行二维DCT变换
        dct_coeff_flat = np.abs(dct_coeff.flatten())
        dct_coeff_sorted = np.sort(dct_coeff_flat)[::-1]
        ###将二维DCT系数矩阵转换为降序排列的一维数组，便于后续能量分析。

        # 计算保留95%能量所需的系数比例
        total_energy = np.sum(dct_coeff_sorted ** 2)
        cumulative_energy = np.cumsum(dct_coeff_sorted ** 2)
        k_dct = np.argmax(cumulative_energy > 0.95 * total_energy)
        sparsity_dct = k_dct / len(dct_coeff_flat)

        print(f"DCT域稀疏性: {sparsity_dct * 100:.2f}%的系数保留95%能量")
        print(f"非零系数估计: {k_dct}/{len(dct_coeff_flat)}")

        # 3. 压缩感知测量
        n_pixels = h * w  #计算图片的像素总数，一共有h*w个像素点
        m = int(measurement_rate * n_pixels)
        # measurement_rate=0.3,采集30%的数据
        print(f"\n进行压缩感知测量...")
        print(f"原始像素数: {n_pixels}")
        print(f"测量数: {m} (压缩比: {n_pixels / m:.2f}:1)")
        # 生成随机测量矩阵（与DCT基低相干）
        np.random.seed(42)
        Phi = np.random.randn(m, n_pixels) / np.sqrt(m)

        # 向量化图像
        x_original = img_gray.flatten()

        # 进行测量
        y = Phi @ x_original

        # 4. 基于稀疏性的重构
        print("\n进行稀疏性重构...")

        # 方法1: DCT基重构
        def reconstruct_dct_basis(y, Phi, h, w, alpha=0.001):
            """使用DCT基进行ℓ1最小化重构"""
            n = h * w

            # 构建DCT基矩阵（简化版，使用分块处理避免内存溢出）
            block_size = min(32, n)
            dct_basis = np.zeros((n, block_size))

            # 分块处理
            x_recon = np.zeros(n)
            for i in range(0, n, block_size):
                end_idx = min(i + block_size, n)
                current_block_size = end_idx - i

                # 构建当前块的DCT基
                block_basis = np.zeros((n, current_block_size))
                for j in range(current_block_size):
                    basis_vec = np.zeros(n)
                    basis_vec[i + j] = 1
                    block_basis[:, j] = idct(basis_vec.reshape(h, w), norm='ortho').flatten()

                # 测量矩阵对应块
                A_block = Phi @ block_basis

                # Lasso重构
                lasso = Lasso(alpha=alpha, max_iter=2000, tol=1e-4)
                lasso.fit(A_block, y)

                # 累加重构结果
                x_recon += block_basis @ lasso.coef_

            return x_recon.reshape(h, w)

        # 方法2: 迭代阈值重构（更内存友好）
        def reconstruct_iterative(y, Phi, h, w, n_iter=100):
            """迭代硬阈值重构算法"""
            n = h * w
            x = np.zeros(n)

            for i in range(n_iter):
                # 计算梯度
                residual = y - Phi @ x
                gradient = Phi.T @ residual

                # 梯度上升
                x = x + 0.1 * gradient

                # 硬阈值：保留最大的k个系数
                k = int(0.1 * n)  # 保留10%的最大系数
                threshold = np.sort(np.abs(x))[-k]
                x[np.abs(x) < threshold] = 0

            return x.reshape(h, w)

        # 选择重构方法（根据内存情况）
        try:
            print("尝试DCT基重构...")
            img_recon = reconstruct_dct_basis(y, Phi, h, w)
            method = "DCT基ℓ1最小化"
        except MemoryError:
            print("内存不足，使用迭代阈值重构...")
            img_recon = reconstruct_iterative(y, Phi, h, w)
            method = "迭代硬阈值"

        # 5. 评估重构质量
        mse = np.mean((img_gray - img_recon) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float('inf')

        # 计算结构相似性
        ssim_value = ssim(img_gray, img_recon, data_range=img_gray.max() - img_gray.min())

        print(f"\n重构质量评估:")
        print(f"重构方法: {method}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
        print(f"均方误差: {mse:.6f}")

        # 6. 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('基于稀疏性的图像压缩感知重构', fontsize=16, fontweight='bold')

        # 原始图像
        axes[0, 0].imshow(img_gray, cmap='gray')
        axes[0, 0].set_title('原始图像\n({}x{})'.format(h, w))
        axes[0, 0].axis('off')

        # 重构图像
        axes[0, 1].imshow(img_recon, cmap='gray')
        axes[0, 1].set_title('重构图像\nPSNR: {:.2f}dB'.format(psnr))
        axes[0, 1].axis('off')

        # 误差图
        error = np.abs(img_gray - img_recon)
        axes[0, 2].imshow(error, cmap='hot')
        axes[0, 2].set_title('重构误差\nMSE: {:.6f}'.format(mse))
        axes[0, 2].axis('off')

        # DCT系数分布
        axes[1, 0].plot(dct_coeff_sorted)
        axes[1, 0].set_title('DCT系数衰减曲线')
        axes[1, 0].set_xlabel('系数索引')
        axes[1, 0].set_ylabel('系数幅度')
        axes[1, 0].grid(True)

        # 稀疏性分析
        energy_ratio = cumulative_energy / total_energy
        axes[1, 1].plot(energy_ratio)
        axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95%能量')
        axes[1, 1].axvline(x=k_dct, color='g', linestyle='--', label=f'{sparsity_dct * 100:.1f}%系数')
        axes[1, 1].set_title('能量累积曲线')
        axes[1, 1].set_xlabel('系数数量')
        axes[1, 1].set_ylabel('能量比例')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 压缩感知原理说明
        axes[1, 2].axis('off')
        info_text = f"""压缩感知原理:

测量率: {measurement_rate * 100}%
压缩比: {n_pixels / m:.1f}:1
稀疏度: {sparsity_dct * 100:.1f}%

关键公式:
m ≥ C·μ²·S·log(n)

• m: 测量数 ({m})
• S: 稀疏度 ({k_dct})
• μ: 相干性 (低)
• n: 信号维度 ({n_pixels})

重构方法: {method}"""

        axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()
        plt.show()

        return {
            'original': img_gray,
            'reconstructed': img_recon,
            'psnr': psnr,
            'ssim': ssim_value,
            'sparsity': sparsity_dct,
            'compression_ratio': n_pixels / m
        }

    except Exception as e:
        print(f"错误: {e}")
        print("请检查图片路径是否正确，或尝试安装所需库:")
        print("pip install pillow scikit-learn scikit-image opencv-python")
        return None


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际图片路径
    image_path = r'E:\Picture\wuzang.jpg'

    # 进行压缩感知重构
    result = compressive_sensing_image_reconstruction(image_path, measurement_rate=0.3)

    if result:
        print(f"\n重构完成!")
        print(f"稀疏性: {result['sparsity'] * 100:.2f}%")
        print(f"压缩比: {result['compression_ratio']:.2f}:1")
        print(f"重构质量 - PSNR: {result['psnr']:.2f}dB, SSIM: {result['ssim']:.4f}")