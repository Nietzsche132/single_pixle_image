"""
基于实际光路的 PEM 偏振单像素成像仿真

实际光路：
白光光源 -> DMD -> 起偏器 -> 透镜1放大 -> 物体 -> 透镜2/透镜3的4f缩束系统
-> PEM -> 与起偏器垂直的检偏器 -> 聚光透镜 -> 探测器输出电压
-> 锁相放大器提取 DC/f/2f -> 计算机算法重建。

重要说明：
1. 本代码优先保证结构清晰、可运行、可生成结果图。
2. 代码中的物理参数均为“示例假设”，不是你的真实实验参数。
3. 探测器最终输出量是电压 V(t)，不是电流；中间使用
   光功率 -> 光电流 -> 跨阻增益 -> 电压 的简化模型。
4. 当前默认使用单色等效模型进行可运行仿真；代码中保留 RGB 白光
   有效 PEM 系数估算，用于说明白光会带来的误差来源。
5. 当前正交检偏器 + PEM 快轴 45° 的简化关系为：

       P_det(t) = eta_post / 2 * [S0 - S1*cos(delta(t)) + S3*sin(delta(t))]

   因此：
       DC  -> S0
       f   -> S3
       2f  -> S1

   S2 需要额外偏振分析构型才能完整恢复。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 0. 路径和主要参数：后续修改优先改这里
# ============================================================

# 输入图像路径：用户提供的待成像物体
INPUT_IMAGE_PATH = r"E:\single_pixel\picture\uestc.png"

# 输出目录：按用户要求固定
OUTPUT_DIR = r"E:\single_pixel\仿真"


# -----------------------------
# 图像与 DMD 编码参数
# -----------------------------

# 不再预设 32x32 仿真尺寸。图像高度和宽度在运行时直接从原图读取。
# DMD 编码按 NumPy C-order（逐行）展平，并补零到不小于物理像素数的
# 最小 2 的整数次幂；投影和反演使用 FWHT，不显式创建巨型 Hadamard 矩阵。

# 是否反相输入图像：默认 False，即白色=高透过率，黑色=低透过率。
INVERT_INPUT_IMAGE = False

# 保存多少个 DMD 图案示例
NUM_MASK_EXAMPLES = 8

# 用哪个 DMD 图案生成示例 PEM 电压时间信号
EXAMPLE_PATTERN_INDEX = 5


# -----------------------------
# 光源、DMD、透镜、物体参数
# 全部为示例假设，不是实验标定值
# -----------------------------

SOURCE_POWER_W = 10e-3                 # 入射到 DMD 的光功率，10 mW，示例假设
DMD_REFLECTIVITY = 0.60                # DMD 有效反射效率，示例假设
POLARIZER_TRANSMISSION = 0.90          # 起偏器材料透过率，示例假设
LENS1_TRANSMISSION = 0.95              # 透镜1透过率，示例假设
OBJECT_BASE_TRANSMISSION = 0.85        # 物体整体附加透过率，示例假设
LENS2_TRANSMISSION = 0.95              # 透镜2透过率，示例假设
LENS3_TRANSMISSION = 0.95              # 透镜3透过率，示例假设
ANALYZER_TRANSMISSION = 0.90           # 检偏器材料透过率，示例假设
COLLECTOR_LENS_TRANSMISSION = 0.95     # 聚光透镜透过率，示例假设

# 透镜1放大倍率与4f缩束倍率
LENS1_MAGNIFICATION = 3.0              # 用户指定：透镜1放大倍率为3
FOUR_F_SHRINK_MAGNIFICATION = 0.5      # 用户指定：4f缩束倍率为0.5

# 光束尺寸，用于检查 PEM 口径匹配；示例假设
DMD_PATTERN_DIAMETER_MM = 2.0
PEM_CLEAR_APERTURE_MM = 8.0


# -----------------------------
# 物体偏振响应参数
# -----------------------------

# 示例假设：为了演示 f 与 2f 信号，令物体产生空间变化的相位延迟。
# 若只想模拟普通强度单像素成像，可以把 RETARDANCE_MAX_DEG 改为 0。
RETARDANCE_MAX_DEG = 60.0


# -----------------------------
# PEM 与锁相参数
# -----------------------------

PEM_FREQUENCY_HZ = 50_000.0            # PEM 调制频率，示例假设
PEM_DELTA0_RAD = 2.4048                # 示例假设：使 J0(delta0) 约等于0
PEM_AXIS_DEG = 45.0                    # PEM 快轴相对起偏器方向45°，示例假设

# 为示例时间信号设置采样率和采样周期数
SAMPLE_RATE_HZ = 2_000_000.0           # 示例假设：2 MHz
NUM_PEM_PERIODS_FOR_EXAMPLE = 12       # 示例时间信号包含多少个 PEM 周期


# -----------------------------
# 探测器电压模型参数
# -----------------------------

DETECTOR_RESPONSIVITY_A_PER_W = 0.45   # 光电探测器响应度，A/W，示例假设
TRANSIMPEDANCE_GAIN_V_PER_A = 10_000.0 # 跨阻增益，V/A，示例假设
VOLTAGE_OFFSET_V = 0.0                 # 电压零偏，示例假设


# -----------------------------
# 噪声参数
# -----------------------------

RANDOM_SEED = 7

# 示例假设：锁相提取 DC/f/2f 后，每个分量的等效电压噪声
LOCKIN_COMPONENT_NOISE_STD_V = 1e-4

# 示例假设：示例 V(t) 时间序列中的探测器电压噪声
TIME_SIGNAL_NOISE_STD_V = 2e-4

# 示例假设：每个 DMD 图案测量时的慢光强波动
SOURCE_RELATIVE_NOISE_STD = 0.002


# ============================================================
# 1. 基础数学和图像工具
# ============================================================

def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_font(size: int = 14) -> ImageFont.ImageFont:
    """尽量获取字体；如果系统没有对应字体，则使用默认字体。"""
    for name in ["arial.ttf", "DejaVuSans.ttf", "simhei.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def bessel_j(n: int, x: float, terms: int = 50) -> float:
    """
    用级数计算第一类贝塞尔函数 J_n(x)，避免依赖 scipy。
    对本仿真中 n=0,1,2 且 x≈2.4 的情况精度足够。
    """
    total = 0.0
    for m in range(terms):
        total += ((-1) ** m) * (x / 2) ** (2 * m + n) / (
            math.factorial(m) * math.factorial(m + n)
        )
    return float(total)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """返回大于等于 n 的最小 2 的整数次幂。"""
    if n <= 0:
        raise ValueError("n must be positive.")
    return 1 << (n - 1).bit_length()


def fast_walsh_hadamard_transform(values: np.ndarray) -> np.ndarray:
    """
    计算一维未归一化 Sylvester 型快速 Walsh-Hadamard 变换（FWHT）。

    变换长度必须为 2 的整数次幂。该实现只保存输入向量和每一级蝶形运算
    所需的临时半块，内存复杂度 O(N)，不会创建 N x N Hadamard 矩阵。

    Sylvester Hadamard 矩阵满足 H @ H = N * I，因此同一函数也用于反演；
    反演结果需要再除以 N。
    """
    out = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    n = out.size
    if not is_power_of_two(n):
        raise ValueError(f"FWHT length n={n} is not a power of two.")

    half_width = 1
    while half_width < n:
        blocks = out.reshape(-1, 2 * half_width)
        left = blocks[:, :half_width].copy()
        right = blocks[:, half_width:]
        blocks[:, :half_width] = left + right
        blocks[:, half_width:] = left - right
        half_width *= 2
    return out


def hadamard_project(values: np.ndarray, transform_size: int) -> np.ndarray:
    """
    将物理像素向量尾部补零到 transform_size，然后执行 FWHT。

    输出的第 k 项等价于补零后 Sylvester Hadamard 第 k 行与输入的内积，
    也等价于物理 DMD 互补图案 M+ 和 M- 的测量差。
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not is_power_of_two(transform_size):
        raise ValueError("transform_size must be a power of two.")
    if flat.size > transform_size:
        raise ValueError(
            f"Input has {flat.size} pixels, larger than transform_size={transform_size}."
        )
    padded = np.zeros(transform_size, dtype=np.float64)
    padded[:flat.size] = flat
    return fast_walsh_hadamard_transform(padded)


def hadamard_reconstruct(
    projections: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """由完整 FWHT 投影反演补零向量，裁剪后按 C-order 恢复二维图像。"""
    proj = np.asarray(projections, dtype=np.float64).reshape(-1)
    if not is_power_of_two(proj.size):
        raise ValueError("The number of projections must be a power of two.")

    n_output = int(np.prod(output_shape))
    if n_output > proj.size:
        raise ValueError(
            f"output_shape contains {n_output} pixels, but only {proj.size} projections exist."
        )
    padded_reconstruction = fast_walsh_hadamard_transform(proj) / proj.size
    return padded_reconstruction[:n_output].reshape(output_shape, order="C")


def sylvester_hadamard_row(
    row_index: int,
    transform_size: int,
    output_length: int,
) -> np.ndarray:
    """
    按需生成一条 Sylvester Hadamard 行的前 output_length 个元素。

    H[k, j] = (-1)**popcount(k & j)。该函数只用于保存少量 DMD 示例和
    生成一个示例时域信号，不用于生成完整矩阵。
    """
    if not is_power_of_two(transform_size):
        raise ValueError("transform_size must be a power of two.")
    if not 0 <= row_index < transform_size:
        raise IndexError(f"row_index={row_index} is outside [0, {transform_size}).")
    if not 0 < output_length <= transform_size:
        raise ValueError("output_length must be in [1, transform_size].")

    columns = np.arange(output_length, dtype=np.uint64)
    parity = np.zeros(output_length, dtype=np.uint8)
    active_bits = int(row_index)
    bit_position = 0
    while active_bits:
        if active_bits & 1:
            parity ^= ((columns >> bit_position) & 1).astype(np.uint8)
        active_bits >>= 1
        bit_position += 1
    return 1 - 2 * parity.astype(np.int8)


def resize_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return img.resize(size, resample)


def load_input_image(path: str, invert: bool = False) -> np.ndarray:
    """
    按原始像素尺寸读取输入图片，转为灰度并归一化到 [0,1]。
    这里将灰度值解释为物体强度透过率：白色=高透过，黑色=低透过。
    """
    with Image.open(path) as source:
        arr = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr
    return np.clip(arr, 0.0, 1.0)


def save_gray_image(arr: np.ndarray, path: str | Path, normalize: bool = False) -> None:
    """保存灰度图。normalize=True 时按 min-max 拉伸显示。"""
    x = np.asarray(arr, dtype=np.float64)
    if normalize:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        else:
            x = np.zeros_like(x)
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray((x * 255).astype(np.uint8), mode="L").save(path)


def gaussian_filter_2d(arr: np.ndarray, sigma: float = 1.5,
                       window_size: int = 11) -> np.ndarray:
    """使用 NumPy 实现反射边界的二维可分离高斯滤波。"""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("gaussian_filter_2d expects a 2D array.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    radius = window_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)

    padded_x = np.pad(x, ((0, 0), (radius, radius)), mode="reflect")
    filtered_x = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x
    )
    padded_y = np.pad(filtered_x, ((radius, radius), (0, 0)), mode="reflect")
    return np.apply_along_axis(
        lambda column: np.convolve(column, kernel, mode="valid"), 0, padded_y
    )


def structural_similarity_index(
    reference: np.ndarray,
    estimate: np.ndarray,
    data_range: float = 1.0,
    sigma: float = 1.5,
    window_size: int = 11,
) -> float:
    """
    计算局部高斯窗口 SSIM。

    采用 11x11、sigma=1.5、K1=0.01、K2=0.03 的常用设置，避免额外依赖
    scikit-image。输入透过率图的 data_range 固定为 1。
    """
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("SSIM inputs must be 2D arrays with identical shapes.")
    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    mu_x = gaussian_filter_2d(x, sigma=sigma, window_size=window_size)
    mu_y = gaussian_filter_2d(y, sigma=sigma, window_size=window_size)
    sigma_x2 = np.maximum(
        gaussian_filter_2d(x * x, sigma=sigma, window_size=window_size) - mu_x * mu_x,
        0.0,
    )
    sigma_y2 = np.maximum(
        gaussian_filter_2d(y * y, sigma=sigma, window_size=window_size) - mu_y * mu_y,
        0.0,
    )
    sigma_xy = (
        gaussian_filter_2d(x * y, sigma=sigma, window_size=window_size) - mu_x * mu_y
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / np.maximum(denominator, np.finfo(np.float64).tiny)
    return float(np.clip(np.mean(ssim_map), -1.0, 1.0))


def pearson_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    """稳健计算两幅图像展平后的 Pearson 相关系数。"""
    x = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(estimate, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("Correlation inputs must contain the same number of values.")

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= np.finfo(np.float64).tiny:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.clip(np.dot(x_centered, y_centered) / denominator, -1.0, 1.0))


def compute_transmittance_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, float]:
    """计算归一化透过率图的定量重建指标。"""
    ref = np.asarray(reference, dtype=np.float64)
    rec = np.asarray(reconstruction, dtype=np.float64)
    if ref.shape != rec.shape:
        raise ValueError("Transmittance reference and reconstruction shapes do not match.")

    error = rec - ref
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    psnr = float("inf") if rmse == 0.0 else float(20.0 * math.log10(1.0 / rmse))
    return {
        "rmse": rmse,
        "mae": mae,
        "psnr_db": psnr,
        "ssim": structural_similarity_index(ref, rec, data_range=1.0),
        "max_absolute_error": float(np.max(np.abs(error))),
        "pearson_correlation": pearson_correlation(ref, rec),
    }


def compute_stokes_component_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, float]:
    """计算单个 Stokes 分量的有量纲与归一化误差指标。"""
    ref = np.asarray(reference, dtype=np.float64)
    rec = np.asarray(reconstruction, dtype=np.float64)
    if ref.shape != rec.shape:
        raise ValueError("Stokes reference and reconstruction shapes do not match.")

    error = rec - ref
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    full_scale = float(np.max(np.abs(ref)))
    ref_l2 = float(np.linalg.norm(ref.ravel()))
    error_l2 = float(np.linalg.norm(error.ravel()))
    eps = np.finfo(np.float64).tiny
    return {
        "rmse_W_per_pixel": rmse,
        "mae_W_per_pixel": mae,
        "nrmse_by_gt_peak": rmse / max(full_scale, eps),
        "relative_l2_error": error_l2 / max(ref_l2, eps),
        "pearson_correlation": pearson_correlation(ref, rec),
        "gt_peak_abs_W_per_pixel": full_scale,
    }


def summarize_integrated_stokes(
    s0: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    s3: np.ndarray,
) -> dict[str, object]:
    """汇总整幅图像的积分 Stokes 向量及其归一化形式。"""
    totals = {
        "S0_W": float(np.sum(s0)),
        "S1_W": float(np.sum(s1)),
        "S2_W": float(np.sum(s2)),
        "S3_W": float(np.sum(s3)),
    }
    s0_total = totals["S0_W"]
    if abs(s0_total) <= np.finfo(np.float64).tiny:
        normalized = {"s0": float("nan"), "s1": float("nan"),
                      "s2": float("nan"), "s3": float("nan")}
        dop = float("nan")
    else:
        normalized = {
            "s0": 1.0,
            "s1": totals["S1_W"] / s0_total,
            "s2": totals["S2_W"] / s0_total,
            "s3": totals["S3_W"] / s0_total,
        }
        dop = float(math.sqrt(
            normalized["s1"] ** 2
            + normalized["s2"] ** 2
            + normalized["s3"] ** 2
        ))
    return {
        "total_stokes_W": totals,
        "normalized_stokes": normalized,
        "integrated_degree_of_polarization": dop,
    }


def arr_to_labeled_image(arr: np.ndarray, label: str, size_px: int = 180,
                         normalize: bool = False) -> Image.Image:
    """把二维数组转成带标签的小图，用于拼接展示。"""
    x = np.asarray(arr, dtype=np.float64)
    if normalize:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        else:
            x = np.zeros_like(x)
    x = np.clip(x, 0.0, 1.0)
    im = Image.fromarray((x * 255).astype(np.uint8), mode="L").convert("RGB")
    im = resize_image(im, (size_px, size_px))
    canvas = Image.new("RGB", (size_px, size_px + 28), "white")
    canvas.paste(im, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), label, fill=(0, 0, 0), font=get_font(13))
    return canvas


def make_contact_sheet(images: list[Image.Image], cols: int, pad: int = 12,
                       bg: str = "white") -> Image.Image:
    rows = int(math.ceil(len(images) / cols))
    w, h = images[0].size
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad,
                              rows * h + (rows + 1) * pad), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        sheet.paste(im, (x, y))
    return sheet


def draw_line_plot(
    series: list[tuple[str, np.ndarray]],
    path: str | Path,
    title: str,
    x_label: str = "index",
    y_label: str = "value",
    width: int = 1000,
    height: int = 520,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    """用 PIL 画简单折线图，避免依赖 matplotlib。"""
    if colors is None:
        colors = [(40, 90, 220), (220, 60, 60), (30, 160, 70), (150, 80, 180)]

    arrays = [np.asarray(y, dtype=np.float64).ravel() for _, y in series]
    max_len = max(len(y) for y in arrays)
    y_min = min(float(np.min(y)) for y in arrays)
    y_max = max(float(np.max(y)) for y in arrays)
    if y_max == y_min:
        y_max = y_min + 1.0
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    font = get_font(14)
    draw.text((margin_l, 18), title, fill=(0, 0, 0), font=get_font(18))

    # 坐标轴
    x0, y0 = margin_l, margin_t + plot_h
    draw.line([x0, margin_t, x0, y0], fill=(0, 0, 0), width=2)
    draw.line([x0, y0, margin_l + plot_w, y0], fill=(0, 0, 0), width=2)
    draw.text((width // 2 - 30, height - 36), x_label, fill=(0, 0, 0), font=font)
    draw.text((12, margin_t + 10), y_label, fill=(0, 0, 0), font=font)

    # y 轴刻度
    for i in range(5):
        val = y_min + (y_max - y_min) * i / 4
        yy = y0 - int((val - y_min) / (y_max - y_min) * plot_h)
        draw.line([x0 - 5, yy, x0, yy], fill=(0, 0, 0))
        draw.text((8, yy - 8), f"{val:.3g}", fill=(0, 0, 0), font=get_font(12))

    # 曲线
    for idx, (name, y) in enumerate(series):
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if n == 1:
            continue

        # 600x600 全采样时测量向量有 524288 项。绘图只需保留与屏幕
        # 分辨率相当的代表点，避免创建数十万个 PIL 坐标元组。
        max_plot_points = max(2, 2 * plot_w)
        if n > max_plot_points:
            sample_indices = np.linspace(0, n - 1, max_plot_points, dtype=np.int64)
            y_plot = y[sample_indices]
        else:
            sample_indices = np.arange(n, dtype=np.int64)
            y_plot = y

        pts = []
        for j, val in zip(sample_indices, y_plot):
            xx = margin_l + int(j / (max_len - 1) * plot_w)
            yy = y0 - int((val - y_min) / (y_max - y_min) * plot_h)
            pts.append((xx, yy))
        draw.line(pts, fill=colors[idx % len(colors)], width=2)
        lx = margin_l + 15
        ly = margin_t + 18 + idx * 22
        draw.line([lx, ly + 7, lx + 24, ly + 7], fill=colors[idx % len(colors)], width=3)
        draw.text((lx + 32, ly), name, fill=(0, 0, 0), font=font)

    im.save(path)


def draw_spectrum(freqs: np.ndarray, amp: np.ndarray, path: str | Path,
                  max_freq_hz: float = 200_000.0) -> None:
    """保存电压信号频谱图。"""
    mask = freqs <= max_freq_hz
    f = freqs[mask]
    a = amp[mask]
    # 避免 DC 太大淹没其他谐波，用线性幅值展示，不做 dB。
    draw_line_plot(
        [("FFT amplitude", a)],
        path,
        title="Voltage signal spectrum",
        x_label=f"frequency bins up to {max_freq_hz/1000:.0f} kHz",
        y_label="amplitude / V",
        width=1000,
        height=520,
    )


# ============================================================
# 2. 实际光路对应的物理模块
# ============================================================

def build_object_stokes(transmittance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造物体后的 Stokes 分布。

    实际光路对应：
    白光 -> DMD -> 起偏器 -> 透镜1放大 -> 物体 -> 4f缩束 -> PEM

    简化假设：
    1. 起偏器输出水平线偏振；
    2. 物体灰度表示强度透过率；
    3. 为了演示 PEM 的 f/2f 分量，额外假设物体引入与灰度相关的相位延迟；
    4. 物体等效为快轴45°的线性延迟器，因此：
           S1 = S0*cos(Gamma)
           S3 = S0*sin(Gamma)
       S2 在当前单构型中不重建。
    """
    n_pix = transmittance.size

    # 光源 -> DMD -> 起偏器 -> 透镜1
    # 非偏振光经过理想起偏器损失一半，然后乘起偏器材料透过率。
    full_on_power_after_lens1 = (
        SOURCE_POWER_W
        * DMD_REFLECTIVITY
        * 0.5
        * POLARIZER_TRANSMISSION
        * LENS1_TRANSMISSION
    )

    # 假设 DMD 全开时，功率均匀分布到输入图像的全部物理像素。
    power_per_pixel_before_object = full_on_power_after_lens1 / n_pix

    # 物体透过率 + 4f系统透过率
    transmission_after_object_and_4f = (
        OBJECT_BASE_TRANSMISSION
        * LENS2_TRANSMISSION
        * LENS3_TRANSMISSION
    )

    s0 = power_per_pixel_before_object * transmittance * transmission_after_object_and_4f

    # 示例偏振响应：相位延迟随灰度变化
    gamma = np.deg2rad(RETARDANCE_MAX_DEG) * transmittance
    s1 = s0 * np.cos(gamma)
    s2 = np.zeros_like(s0)
    s3 = s0 * np.sin(gamma)
    return s0, s1, s2, s3


def pem_analyzer_power(
    s0: float | np.ndarray,
    s1: float | np.ndarray,
    s3: float | np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    PEM + 正交检偏器后的光功率。

    实际器件对应：
    PEM 引入 delta(t)，正交检偏器把偏振态变化投影为强度变化。

    简化模型：
        P(t) = eta_post/2 * [S0 - S1*cos(delta(t)) + S3*sin(delta(t))]

    其中 eta_post 包含检偏器材料透过率和聚光透镜透过率。
    """
    delta = PEM_DELTA0_RAD * np.sin(2 * np.pi * PEM_FREQUENCY_HZ * t)
    eta_post = ANALYZER_TRANSMISSION * COLLECTOR_LENS_TRANSMISSION
    power = 0.5 * eta_post * (s0 - s1 * np.cos(delta) + s3 * np.sin(delta))
    return np.maximum(power, 0.0)


def optical_power_to_voltage(power_w: np.ndarray) -> np.ndarray:
    """
    探测器模型：光功率 -> 光电流 -> 电压。

    i(t) = R * P(t)
    V(t) = G_TIA * i(t) + V_offset

    注意：最终进入锁相放大器的是电压 V(t)，不是电流。
    """
    photocurrent_a = DETECTOR_RESPONSIVITY_A_PER_W * power_w
    voltage_v = TRANSIMPEDANCE_GAIN_V_PER_A * photocurrent_a + VOLTAGE_OFFSET_V
    return voltage_v


def lockin_components_from_projections(
    s0_proj: np.ndarray,
    s1_proj: np.ndarray,
    s3_proj: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个 DMD Hadamard 投影，直接计算锁相应提取到的电压 DC/f/2f 分量。

    这里使用解析谐波系数，避免为每个 DMD 图案都生成完整 V(t)，提高速度。

    对应关系：
        V_DC = G * eta_post/2 * (S0 - J0*S1)
        V_f  = G * eta_post   * J1*S3
        V_2f = G * eta_post  * (-J2*S1)

    其中 G = detector_responsivity * transimpedance_gain。
    """
    j0 = bessel_j(0, PEM_DELTA0_RAD)
    j1 = bessel_j(1, PEM_DELTA0_RAD)
    j2 = bessel_j(2, PEM_DELTA0_RAD)

    eta_post = ANALYZER_TRANSMISSION * COLLECTOR_LENS_TRANSMISSION
    voltage_gain_v_per_w = DETECTOR_RESPONSIVITY_A_PER_W * TRANSIMPEDANCE_GAIN_V_PER_A

    # 示例假设：慢光强波动，对每个DMD图案有一个乘性因子。
    source_factor = 1.0 + rng.normal(0.0, SOURCE_RELATIVE_NOISE_STD, size=s0_proj.shape)

    s0_eff = s0_proj * source_factor
    s1_eff = s1_proj * source_factor
    s3_eff = s3_proj * source_factor

    v_dc = voltage_gain_v_per_w * eta_post * 0.5 * (s0_eff - j0 * s1_eff)
    v_f = voltage_gain_v_per_w * eta_post * (j1 * s3_eff)
    v_2f = voltage_gain_v_per_w * eta_post * (-j2 * s1_eff)

    # 示例假设：锁相分量测量噪声，单位 V
    if LOCKIN_COMPONENT_NOISE_STD_V > 0:
        v_dc = v_dc + rng.normal(0.0, LOCKIN_COMPONENT_NOISE_STD_V, size=v_dc.shape)
        v_f = v_f + rng.normal(0.0, LOCKIN_COMPONENT_NOISE_STD_V, size=v_f.shape)
        v_2f = v_2f + rng.normal(0.0, LOCKIN_COMPONENT_NOISE_STD_V, size=v_2f.shape)

    return v_dc, v_f, v_2f


def recover_stokes_projections_from_lockin(
    v_dc: np.ndarray,
    v_f: np.ndarray,
    v_2f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    由锁相电压分量恢复每个 DMD 图案下的 Stokes 投影。

    注意：
    当前单构型恢复 S0, S1, S3；S2 需要额外偏振分析构型。
    """
    j0 = bessel_j(0, PEM_DELTA0_RAD)
    j1 = bessel_j(1, PEM_DELTA0_RAD)
    j2 = bessel_j(2, PEM_DELTA0_RAD)

    eta_post = ANALYZER_TRANSMISSION * COLLECTOR_LENS_TRANSMISSION
    voltage_gain_v_per_w = DETECTOR_RESPONSIVITY_A_PER_W * TRANSIMPEDANCE_GAIN_V_PER_A
    scale = voltage_gain_v_per_w * eta_post

    s1_hat = -v_2f / (scale * j2)
    s3_hat = v_f / (scale * j1)
    s0_hat = 2 * v_dc / scale + j0 * s1_hat
    return s0_hat, s1_hat, s3_hat


def numerical_lockin(signal_v: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """
    对示例 V(t) 做数值锁相。
    用于输出报告，不用于整幅图像的快速重建。
    """
    w = 2 * np.pi * PEM_FREQUENCY_HZ
    dc = float(np.mean(signal_v))
    f_sin = float(2 * np.mean(signal_v * np.sin(w * t)))
    f_cos = float(2 * np.mean(signal_v * np.cos(w * t)))
    f_amp = float(math.sqrt(f_sin ** 2 + f_cos ** 2))
    f_phase = float(math.atan2(f_sin, f_cos))

    twof_sin = float(2 * np.mean(signal_v * np.sin(2 * w * t)))
    twof_cos = float(2 * np.mean(signal_v * np.cos(2 * w * t)))
    twof_amp = float(math.sqrt(twof_sin ** 2 + twof_cos ** 2))
    twof_phase = float(math.atan2(twof_sin, twof_cos))

    return {
        "dc_v": dc,
        "f_sin_v": f_sin,
        "f_cos_v": f_cos,
        "f_amp_v": f_amp,
        "f_phase_rad": f_phase,
        "twof_sin_v": twof_sin,
        "twof_cos_v": twof_cos,
        "twof_amp_v": twof_amp,
        "twof_phase_rad": twof_phase,
    }


# ============================================================
# 3. 可视化输出
# ============================================================

def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """以给定 x 坐标为中心绘制一行文本。"""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    draw.text((xy[0] - text_width // 2, xy[1]), text, fill=fill, font=font)


def colorize_scalar_map(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
) -> Image.Image:
    """把标量数组转换为灰度或蓝-白-红发散色图。"""
    x = np.asarray(arr, dtype=np.float64)
    if vmax <= vmin:
        normalized = np.zeros_like(x)
    else:
        normalized = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)

    if cmap == "gray":
        gray = np.rint(255.0 * normalized).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    elif cmap == "diverging":
        negative_color = np.array([49.0, 54.0, 149.0])
        zero_color = np.array([255.0, 255.0, 255.0])
        positive_color = np.array([165.0, 0.0, 38.0])
        rgb = np.empty(x.shape + (3,), dtype=np.float64)
        negative = normalized <= 0.5
        negative_weight = (2.0 * normalized)[..., None]
        positive_weight = (2.0 * normalized - 1.0)[..., None]
        rgb[negative] = (
            negative_color * (1.0 - negative_weight[negative])
            + zero_color * negative_weight[negative]
        )
        rgb[~negative] = (
            zero_color * (1.0 - positive_weight[~negative])
            + positive_color * positive_weight[~negative]
        )
        rgb = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported color map: {cmap}")
    return Image.fromarray(rgb, mode="RGB")


def render_heatmap_panel(
    arr: np.ndarray,
    label: str,
    vmin: float,
    vmax: float,
    unit: str,
    cmap: str = "gray",
    image_size: int = 260,
) -> Image.Image:
    """生成带标题、色条、数值范围和单位的标量图面板。"""
    panel_width = image_size + 60
    panel_height = image_size + 108
    canvas = Image.new("RGB", (panel_width, panel_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_centered_text(draw, (panel_width // 2, 7), label, get_font(16))

    heatmap = colorize_scalar_map(arr, vmin, vmax, cmap)
    heatmap = resize_image(heatmap, (image_size, image_size))
    image_x = (panel_width - image_size) // 2
    image_y = 34
    canvas.paste(heatmap, (image_x, image_y))
    draw.rectangle(
        [image_x, image_y, image_x + image_size - 1, image_y + image_size - 1],
        outline=(80, 80, 80),
        width=1,
    )

    colorbar_width = image_size
    gradient = np.linspace(vmin, vmax, colorbar_width, dtype=np.float64)[None, :]
    colorbar = colorize_scalar_map(gradient, vmin, vmax, cmap).resize((colorbar_width, 14))
    colorbar_y = image_y + image_size + 10
    canvas.paste(colorbar, (image_x, colorbar_y))
    draw.rectangle(
        [image_x, colorbar_y, image_x + colorbar_width - 1, colorbar_y + 13],
        outline=(80, 80, 80),
        width=1,
    )

    number_font = get_font(12)
    min_text = f"{vmin:.4g}"
    max_text = f"{vmax:.4g}"
    draw.text((image_x, colorbar_y + 18), min_text, fill=(0, 0, 0), font=number_font)
    max_bbox = draw.textbbox((0, 0), max_text, font=number_font)
    draw.text(
        (image_x + colorbar_width - (max_bbox[2] - max_bbox[0]), colorbar_y + 18),
        max_text,
        fill=(0, 0, 0),
        font=number_font,
    )
    draw_centered_text(draw, (panel_width // 2, colorbar_y + 18), unit, number_font)
    return canvas


def save_transmittance_reconstruction_summary(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    metrics: dict[str, float],
    path: Path,
) -> None:
    """保存透过率真值、重构、误差及全部定量指标。"""
    absolute_error = np.abs(reconstruction - reference)
    error_max = max(float(metrics["max_absolute_error"]), 1e-12)
    panels = [
        render_heatmap_panel(reference, "Ground truth T", 0.0, 1.0, "transmittance"),
        render_heatmap_panel(reconstruction, "Reconstructed T", 0.0, 1.0, "transmittance"),
        render_heatmap_panel(absolute_error, "Absolute error |T_rec - T|",
                             0.0, error_max, "absolute error"),
    ]

    pad = 18
    title_height = 78
    metrics_height = 112
    width = sum(panel.width for panel in panels) + (len(panels) + 1) * pad
    height = title_height + panels[0].height + metrics_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), "Transmittance reconstruction quality", fill=(0, 0, 0),
              font=get_font(22))
    draw.text(
        (pad, 45),
        "Ground truth and reconstruction share the fixed range [0, 1].",
        fill=(50, 50, 50),
        font=get_font(14),
    )

    x = pad
    for panel in panels:
        canvas.paste(panel, (x, title_height))
        x += panel.width + pad

    y = title_height + panels[0].height + 10
    metric_font = get_font(16)
    draw.text(
        (pad, y),
        f"RMSE = {metrics['rmse']:.6g}    MAE = {metrics['mae']:.6g}    "
        f"Max |error| = {metrics['max_absolute_error']:.6g}",
        fill=(0, 0, 0),
        font=metric_font,
    )
    draw.text(
        (pad, y + 30),
        f"PSNR = {metrics['psnr_db']:.3f} dB    SSIM = {metrics['ssim']:.6f}    "
        f"Pearson r = {metrics['pearson_correlation']:.6f}",
        fill=(0, 0, 0),
        font=metric_font,
    )
    draw.text(
        (pad, y + 61),
        "SSIM: Gaussian 11 x 11 window, sigma = 1.5, data range = 1.",
        fill=(70, 70, 70),
        font=get_font(13),
    )
    canvas.save(path)


def save_ground_truth_stokes_maps(
    s0_gt: np.ndarray,
    s1_gt: np.ndarray,
    s3_gt: np.ndarray,
    path: Path,
) -> None:
    """保存带统一物理单位和共同绝对幅度尺度的初始 Stokes 图。"""
    s0_nw = np.asarray(s0_gt) * 1e9
    s1_nw = np.asarray(s1_gt) * 1e9
    s3_nw = np.asarray(s3_gt) * 1e9
    common_scale = max(
        float(np.max(np.abs(s0_nw))),
        float(np.max(np.abs(s1_nw))),
        float(np.max(np.abs(s3_nw))),
        1e-12,
    )
    panels = [
        render_heatmap_panel(s0_nw, "Ground truth S0", 0.0, common_scale,
                             "nW / pixel", "gray"),
        render_heatmap_panel(s1_nw, "Ground truth S1", -common_scale, common_scale,
                             "nW / pixel", "diverging"),
        render_heatmap_panel(s3_nw, "Ground truth S3", -common_scale, common_scale,
                             "nW / pixel", "diverging"),
    ]
    pad = 18
    title_height = 78
    width = sum(panel.width for panel in panels) + (len(panels) + 1) * pad
    height = title_height + panels[0].height + 18
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), "Ground-truth spatial Stokes parameters", fill=(0, 0, 0),
              font=get_font(22))
    draw.text((pad, 45), "All panels use the same absolute scale; S1 and S3 retain sign.",
              fill=(50, 50, 50), font=get_font(14))
    x = pad
    for panel in panels:
        canvas.paste(panel, (x, title_height))
        x += panel.width + pad
    canvas.save(path)

def save_dmd_mask_examples(
    transform_size: int,
    out_dir: Path,
    image_shape: tuple[int, int],
) -> None:
    """按需生成并保存若干物理尺寸的 DMD 互补正图案 M+。"""
    n_pixels = int(np.prod(image_shape))
    imgs = []
    for k in range(min(NUM_MASK_EXAMPLES, transform_size)):
        signed_row = sylvester_hadamard_row(k, transform_size, n_pixels)
        mask_plus = (signed_row > 0).astype(np.float64).reshape(image_shape, order="C")
        imgs.append(arr_to_labeled_image(mask_plus, f"DMD M+ #{k}", size_px=160))
    sheet = make_contact_sheet(imgs, cols=4)
    sheet.save(out_dir / "02_dmd_mask_examples.png")


def save_stokes_contact_sheet(
    s0_gt: np.ndarray, s1_gt: np.ndarray, s3_gt: np.ndarray,
    s0_rec: np.ndarray, s1_rec: np.ndarray, s3_rec: np.ndarray,
    component_metrics: dict[str, dict[str, float]],
    out_dir: Path,
) -> None:
    """用共同色标比较 Stokes 真值/重构，并显示逐分量指标。"""
    components = [
        ("S0", s0_gt, s0_rec, "gray"),
        ("S1", s1_gt, s1_rec, "diverging"),
        ("S3", s3_gt, s3_rec, "diverging"),
    ]
    rows: list[tuple[Image.Image, Image.Image, str, float, float]] = []
    for name, ground_truth, reconstruction, cmap in components:
        gt_nw = np.asarray(ground_truth) * 1e9
        rec_nw = np.asarray(reconstruction) * 1e9
        if name == "S0":
            vmin = 0.0
            vmax = max(float(np.max(gt_nw)), float(np.max(rec_nw)), 1e-12)
        else:
            scale = max(float(np.max(np.abs(gt_nw))),
                        float(np.max(np.abs(rec_nw))), 1e-12)
            vmin, vmax = -scale, scale
        gt_panel = render_heatmap_panel(gt_nw, f"Ground truth {name}", vmin, vmax,
                                        "nW / pixel", cmap)
        rec_panel = render_heatmap_panel(rec_nw, f"Reconstructed {name}", vmin, vmax,
                                         "nW / pixel", cmap)
        rows.append((gt_panel, rec_panel, name, vmin, vmax))

    pad = 18
    title_height = 88
    metric_width = 360
    panel_width = rows[0][0].width
    panel_height = rows[0][0].height
    width = 4 * pad + 2 * panel_width + metric_width
    height = title_height + len(rows) * panel_height + (len(rows) - 1) * pad + 48
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), "Stokes reconstruction with shared physical scales",
              fill=(0, 0, 0), font=get_font(22))
    draw.text(
        (pad, 45),
        "Each ground-truth/reconstruction pair uses exactly the same color scale.",
        fill=(50, 50, 50),
        font=get_font(14),
    )

    y = title_height
    for gt_panel, rec_panel, name, vmin, vmax in rows:
        canvas.paste(gt_panel, (pad, y))
        canvas.paste(rec_panel, (2 * pad + panel_width, y))

        metric_x = 3 * pad + 2 * panel_width
        metrics = component_metrics[name]
        draw.rectangle(
            [metric_x, y + 34, metric_x + metric_width - pad, y + panel_height - 34],
            outline=(180, 180, 180),
            width=1,
        )
        text_x = metric_x + 18
        text_y = y + 54
        draw.text((text_x, text_y), f"{name} reconstruction metrics",
                  fill=(0, 0, 0), font=get_font(18))
        metric_lines = [
            f"RMSE: {metrics['rmse_W_per_pixel'] * 1e9:.6g} nW/pixel",
            f"MAE: {metrics['mae_W_per_pixel'] * 1e9:.6g} nW/pixel",
            f"NRMSE / GT peak: {100 * metrics['nrmse_by_gt_peak']:.4f}%",
            f"Relative L2 error: {100 * metrics['relative_l2_error']:.4f}%",
            f"Pearson r: {metrics['pearson_correlation']:.6f}",
            f"Display range: [{vmin:.4g}, {vmax:.4g}] nW/pixel",
        ]
        for line_index, line in enumerate(metric_lines):
            draw.text((text_x, text_y + 42 + 34 * line_index), line,
                      fill=(20, 20, 20), font=get_font(15))
        y += panel_height + pad

    draw.text(
        (pad, height - 30),
        "NRMSE = RMSE / max(|ground truth|); signed S1 and S3 use a zero-centered diverging scale.",
        fill=(70, 70, 70),
        font=get_font(13),
    )
    canvas.save(out_dir / "08_stokes_reconstruction_contact_sheet.png")


def save_stokes_quantitative_summary(
    ground_truth_summary: dict[str, object],
    reconstruction_summary: dict[str, object],
    image_shape: tuple[int, int],
    path: Path,
) -> None:
    """保存整幅图像积分及归一化 Stokes 参数对比表。"""
    gt_total = ground_truth_summary["total_stokes_W"]
    rec_total = reconstruction_summary["total_stokes_W"]
    gt_norm = ground_truth_summary["normalized_stokes"]
    rec_norm = reconstruction_summary["normalized_stokes"]

    rows: list[tuple[str, float, float, str]] = []
    for component in ["S0", "S1", "S2", "S3"]:
        rows.append((
            f"sum {component} / mW",
            float(gt_total[f"{component}_W"]) * 1e3,
            float(rec_total[f"{component}_W"]) * 1e3,
            "relative",
        ))
    for component in ["s0", "s1", "s2", "s3"]:
        rows.append((
            f"{component} = sum {component.upper()} / sum S0",
            float(gt_norm[component]),
            float(rec_norm[component]),
            "absolute",
        ))
    rows.append((
        "Integrated DoP",
        float(ground_truth_summary["integrated_degree_of_polarization"]),
        float(reconstruction_summary["integrated_degree_of_polarization"]),
        "absolute",
    ))

    width = 1180
    top = 116
    row_height = 44
    header_height = 46
    bottom_note = 92
    height = top + header_height + row_height * len(rows) + bottom_note
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((34, 18), "Image-integrated Stokes parameters", fill=(0, 0, 0),
              font=get_font(24))
    draw.text(
        (34, 54),
        f"Spatial sum over {image_shape[0]} x {image_shape[1]} pixels; each pixel has its own Stokes vector.",
        fill=(40, 40, 40),
        font=get_font(15),
    )
    draw.text(
        (34, 80),
        "Normalized values divide the integrated S1, S2 and S3 by integrated S0.",
        fill=(70, 70, 70),
        font=get_font(14),
    )

    x_edges = [34, 420, 665, 910, width - 34]
    header_y = top
    headers = ["Quantity", "Ground truth", "Reconstruction", "Difference"]
    for column, header in enumerate(headers):
        draw.text((x_edges[column] + 12, header_y + 12), header,
                  fill=(0, 0, 0), font=get_font(16))
    draw.line([x_edges[0], header_y + header_height,
               x_edges[-1], header_y + header_height], fill=(80, 80, 80), width=2)

    y = header_y + header_height
    for row_index, (label, gt_value, rec_value, difference_type) in enumerate(rows):
        if row_index in (4, 8):
            draw.line([x_edges[0], y, x_edges[-1], y], fill=(120, 120, 120), width=2)
        draw.text((x_edges[0] + 12, y + 12), label, fill=(0, 0, 0), font=get_font(15))
        draw.text((x_edges[1] + 12, y + 12), f"{gt_value:.9g}",
                  fill=(0, 0, 0), font=get_font(15))
        draw.text((x_edges[2] + 12, y + 12), f"{rec_value:.9g}",
                  fill=(0, 0, 0), font=get_font(15))
        if difference_type == "relative":
            if abs(gt_value) <= 1e-30:
                difference = "both zero" if abs(rec_value) <= 1e-30 else "undefined"
            else:
                difference = f"{100.0 * (rec_value - gt_value) / abs(gt_value):+.6g}%"
        else:
            difference = f"{rec_value - gt_value:+.9g}"
        draw.text((x_edges[3] + 12, y + 12), difference,
                  fill=(0, 0, 0), font=get_font(15))
        y += row_height
        draw.line([x_edges[0], y, x_edges[-1], y], fill=(220, 220, 220), width=1)

    for x_edge in x_edges:
        draw.line([x_edge, header_y, x_edge, y], fill=(190, 190, 190), width=1)

    draw.text(
        (34, y + 20),
        "S2 is fixed to zero by the present object model and is not independently reconstructed.",
        fill=(60, 60, 60),
        font=get_font(14),
    )
    draw.text(
        (34, y + 48),
        "Integrated DoP may be below 1 because spatially different Stokes vectors are summed incoherently.",
        fill=(60, 60, 60),
        font=get_font(14),
    )
    canvas.save(path)


def save_csv_measurements(path: Path, v_dc: np.ndarray, v_f: np.ndarray, v_2f: np.ndarray) -> None:
    """保存锁相分量测量序列，便于后续检查。"""
    data = np.column_stack([np.arange(len(v_dc)), v_dc, v_f, v_2f])
    header = "pattern_index,V_DC_V,V_f_V,V_2f_V"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10e")


# ============================================================
# 4. 白光 RGB 简化估算
# ============================================================

def white_light_rgb_effective_coefficients() -> dict[str, object]:
    """
    用 RGB 三个波长估算白光下 PEM 贝塞尔系数的有效变化。
    这不是完整白光传播仿真，只用于记录说明：
        delta0(lambda) ≈ delta0(lambda0) * lambda0/lambda
    会导致 J0/J1/J2 随波长变化。
    """
    lambda0_nm = 550.0
    rgb = [
        {"name": "R", "lambda_nm": 650.0, "spectrum_weight": 0.30, "detector_responsivity": 0.42},
        {"name": "G", "lambda_nm": 550.0, "spectrum_weight": 0.40, "detector_responsivity": 0.45},
        {"name": "B", "lambda_nm": 450.0, "spectrum_weight": 0.30, "detector_responsivity": 0.32},
    ]
    raw_weights = []
    for item in rgb:
        raw_weights.append(item["spectrum_weight"] * item["detector_responsivity"])
    raw_weights = np.asarray(raw_weights, dtype=np.float64)
    norm_weights = raw_weights / np.sum(raw_weights)

    rows = []
    j0_eff = 0.0
    j1_eff = 0.0
    j2_eff = 0.0
    for item, q in zip(rgb, norm_weights):
        delta = PEM_DELTA0_RAD * lambda0_nm / item["lambda_nm"]
        j0 = bessel_j(0, delta)
        j1 = bessel_j(1, delta)
        j2 = bessel_j(2, delta)
        rows.append({
            "name": item["name"],
            "lambda_nm": item["lambda_nm"],
            "normalized_weight": float(q),
            "delta0_rad": float(delta),
            "J0": float(j0),
            "J1": float(j1),
            "J2": float(j2),
        })
        j0_eff += q * j0
        j1_eff += q * j1
        j2_eff += q * j2

    return {
        "rows": rows,
        "J0_eff": float(j0_eff),
        "J1_eff": float(j1_eff),
        "J2_eff": float(j2_eff),
        "note": "RGB is only a simplified estimate, not a calibrated white-light model.",
    }


# ============================================================
# 5. 主仿真流程
# ============================================================

def main() -> None:
    out_dir = ensure_output_dir(OUTPUT_DIR)
    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------------
    # 1) 按原始尺寸读取输入图像，作为待成像物体强度透过率
    # --------------------------------------------------------
    transmittance = load_input_image(INPUT_IMAGE_PATH, INVERT_INPUT_IMAGE)
    image_height, image_width = transmittance.shape
    image_shape = (image_height, image_width)
    n_pixels = int(transmittance.size)

    # 600x600 对应 360000 个物理像素，不是 2 的整数次幂。将 C-order
    # 展平向量尾部补零到 524288，再通过 O(N log N) FWHT 完成投影。
    transform_size = next_power_of_two(n_pixels)
    padding_pixels = transform_size - n_pixels

    save_gray_image(transmittance, out_dir / "01_preprocessed_input_transmittance.png")

    # --------------------------------------------------------
    # 2) 构造物体后的 Stokes 分布：S0, S1, S3
    # --------------------------------------------------------
    s0_map, s1_map, _s2_map, s3_map = build_object_stokes(transmittance)

    save_gray_image(s0_map / np.max(s0_map + 1e-30), out_dir / "03_ground_truth_S0_normalized.png")
    save_gray_image(np.abs(s1_map) / np.max(np.abs(s1_map) + 1e-30), out_dir / "03_ground_truth_abs_S1_normalized.png")
    save_gray_image(np.abs(s3_map) / np.max(np.abs(s3_map) + 1e-30), out_dir / "03_ground_truth_abs_S3_normalized.png")
    save_ground_truth_stokes_maps(
        s0_map,
        s1_map,
        s3_map,
        out_dir / "03_ground_truth_stokes_physical_units.png",
    )

    # --------------------------------------------------------
    # 3) 使用补零 FWHT 生成等效 DMD Hadamard 投影
    # --------------------------------------------------------
    # 不创建 transform_size x transform_size 矩阵；仅为展示按需生成少量行。
    save_dmd_mask_examples(transform_size, out_dir, image_shape)

    # 明确使用 C-order：先逐行展平，反演后也按相同顺序 reshape。
    s0_flat = s0_map.reshape(-1, order="C")
    s1_flat = s1_map.reshape(-1, order="C")
    s3_flat = s3_map.reshape(-1, order="C")

    # Hadamard 投影：等效于补零后每一行 DMD 互补图案 M+ - M- 的测量差分。
    # 虚拟补零像素不对应实际 DMD 区域，对任意图案都贡献零光功率。
    s0_proj = hadamard_project(s0_flat, transform_size)
    s1_proj = hadamard_project(s1_flat, transform_size)
    s3_proj = hadamard_project(s3_flat, transform_size)

    # --------------------------------------------------------
    # 4) 对每个 DMD 图案计算锁相提取的电压分量 DC/f/2f
    # --------------------------------------------------------
    v_dc, v_f, v_2f = lockin_components_from_projections(s0_proj, s1_proj, s3_proj, rng)

    save_csv_measurements(out_dir / "06_lockin_components.csv", v_dc, v_f, v_2f)
    draw_line_plot(
        [("V_DC", v_dc), ("V_f", v_f), ("V_2f", v_2f)],
        out_dir / "06_lockin_components_curves.png",
        title="Lock-in extracted voltage components for each DMD pattern",
        x_label="DMD pattern index",
        y_label="voltage / V",
    )

    # --------------------------------------------------------
    # 5) 由锁相电压恢复每个 DMD 图案下的 Stokes 投影
    # --------------------------------------------------------
    s0_hat_proj, s1_hat_proj, s3_hat_proj = recover_stokes_projections_from_lockin(v_dc, v_f, v_2f)

    # --------------------------------------------------------
    # 6) Hadamard 反演重建图像
    # --------------------------------------------------------
    # 对 transform_size 阶 Sylvester 矩阵，H^{-1} = H / transform_size。
    # 反演完整补零向量后，只保留前 n_pixels 项并恢复原始二维尺寸。
    s0_rec = hadamard_reconstruct(s0_hat_proj, image_shape)
    s1_rec = hadamard_reconstruct(s1_hat_proj, image_shape)
    s3_rec = hadamard_reconstruct(s3_hat_proj, image_shape)

    # 将 S0 重建结果换算回物体透过率图像，便于和输入图像比较
    full_on_power_after_lens1 = (
        SOURCE_POWER_W
        * DMD_REFLECTIVITY
        * 0.5
        * POLARIZER_TRANSMISSION
        * LENS1_TRANSMISSION
    )
    power_per_pixel_before_object = full_on_power_after_lens1 / n_pixels
    stokes_scale = (
        power_per_pixel_before_object
        * OBJECT_BASE_TRANSMISSION
        * LENS2_TRANSMISSION
        * LENS3_TRANSMISSION
    )
    transmittance_rec = np.clip(s0_rec / (stokes_scale + 1e-30), 0.0, 1.0)
    error_map = transmittance_rec - transmittance

    # 定量指标必须先于结果图计算，保证图中、JSON 和文本摘要使用同一组数值。
    transmittance_metrics = compute_transmittance_metrics(transmittance, transmittance_rec)
    stokes_component_metrics = {
        "S0": compute_stokes_component_metrics(s0_map, s0_rec),
        "S1": compute_stokes_component_metrics(s1_map, s1_rec),
        "S3": compute_stokes_component_metrics(s3_map, s3_rec),
    }
    s2_rec = np.zeros_like(s0_rec)
    ground_truth_stokes_summary = summarize_integrated_stokes(
        s0_map, s1_map, _s2_map, s3_map
    )
    reconstruction_stokes_summary = summarize_integrated_stokes(
        s0_rec, s1_rec, s2_rec, s3_rec
    )

    save_gray_image(transmittance_rec, out_dir / "07_reconstructed_transmittance.png")
    save_gray_image(np.abs(error_map), out_dir / "07_absolute_error_map.png", normalize=True)
    save_transmittance_reconstruction_summary(
        transmittance,
        transmittance_rec,
        transmittance_metrics,
        out_dir / "07_transmittance_reconstruction_summary.png",
    )
    save_stokes_contact_sheet(
        s0_map,
        s1_map,
        s3_map,
        s0_rec,
        s1_rec,
        s3_rec,
        stokes_component_metrics,
        out_dir,
    )
    save_stokes_quantitative_summary(
        ground_truth_stokes_summary,
        reconstruction_stokes_summary,
        image_shape,
        out_dir / "09_stokes_quantitative_summary.png",
    )

    # --------------------------------------------------------
    # 7) 生成一个实际物理 DMD 正图案 M+ 下的探测器电压 V(t)
    # --------------------------------------------------------
    k_ex = max(0, min(EXAMPLE_PATTERN_INDEX, transform_size - 1))
    signed_example_row = sylvester_hadamard_row(k_ex, transform_size, n_pixels)
    mask_plus = (signed_example_row > 0).astype(np.float64)

    s0_plus = float(mask_plus @ s0_flat)
    s1_plus = float(mask_plus @ s1_flat)
    s3_plus = float(mask_plus @ s3_flat)

    duration_s = NUM_PEM_PERIODS_FOR_EXAMPLE / PEM_FREQUENCY_HZ
    num_samples = int(round(duration_s * SAMPLE_RATE_HZ))
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE_HZ

    p_t = pem_analyzer_power(s0_plus, s1_plus, s3_plus, t)
    v_t_clean = optical_power_to_voltage(p_t)
    v_t = v_t_clean + rng.normal(0.0, TIME_SIGNAL_NOISE_STD_V, size=v_t_clean.shape)

    lockin_example = numerical_lockin(v_t, t)

    draw_line_plot(
        [("V(t)", v_t)],
        out_dir / "04_example_detector_voltage_signal.png",
        title=f"Detector voltage V(t), physical DMD M+ pattern #{k_ex}",
        x_label="sample index",
        y_label="voltage / V",
    )

    # 频谱图
    v_centered = v_t - np.mean(v_t)
    fft = np.fft.rfft(v_centered)
    freqs = np.fft.rfftfreq(len(v_centered), d=1.0 / SAMPLE_RATE_HZ)
    amp = 2 * np.abs(fft) / len(v_centered)
    draw_spectrum(freqs, amp, out_dir / "05_example_voltage_spectrum.png")

    # --------------------------------------------------------
    # 8) 光路尺寸和4f缩束检查
    # --------------------------------------------------------
    object_beam_diameter_mm = LENS1_MAGNIFICATION * DMD_PATTERN_DIAMETER_MM
    pem_input_beam_diameter_mm = FOUR_F_SHRINK_MAGNIFICATION * object_beam_diameter_mm
    pem_aperture_ok = pem_input_beam_diameter_mm <= PEM_CLEAR_APERTURE_MM

    # 面积缩放用于说明光功率密度变化
    area_after_lens1_factor = LENS1_MAGNIFICATION ** 2
    irradiance_after_lens1_relative = 1.0 / area_after_lens1_factor
    area_after_4f_factor = FOUR_F_SHRINK_MAGNIFICATION ** 2
    irradiance_after_4f_relative_to_object = 1.0 / area_after_4f_factor

    # --------------------------------------------------------
    # 9) 误差指标
    # --------------------------------------------------------
    rmse = transmittance_metrics["rmse"]
    mae = transmittance_metrics["mae"]
    psnr = transmittance_metrics["psnr_db"]
    ssim = transmittance_metrics["ssim"]
    max_absolute_error = transmittance_metrics["max_absolute_error"]
    transmittance_correlation = transmittance_metrics["pearson_correlation"]

    s0_rmse = stokes_component_metrics["S0"]["rmse_W_per_pixel"]
    s1_rmse = stokes_component_metrics["S1"]["rmse_W_per_pixel"]
    s3_rmse = stokes_component_metrics["S3"]["rmse_W_per_pixel"]

    # --------------------------------------------------------
    # 10) 保存参数和说明文件
    # --------------------------------------------------------
    white_light_info = white_light_rgb_effective_coefficients()

    params = {
        "input_image_path": INPUT_IMAGE_PATH,
        "output_dir": OUTPUT_DIR,
        "image_height": image_height,
        "image_width": image_width,
        "image_shape": [image_height, image_width],
        "num_physical_pixels": n_pixels,
        "hadamard_encoding": {
            "scheme": "row-major zero-padding plus 1D Sylvester FWHT",
            "flatten_order": "C",
            "transform_size": transform_size,
            "padding_pixels": padding_pixels,
            "padding_fraction_of_transform": padding_pixels / transform_size,
            "signed_pattern_count": transform_size,
            "complementary_dmd_exposure_count": 2 * transform_size,
            "single_float64_vector_bytes": transform_size * np.dtype(np.float64).itemsize,
            "explicit_int8_matrix_bytes_avoided": transform_size * transform_size,
            "explicit_float64_matrix_bytes_avoided": (
                transform_size * transform_size * np.dtype(np.float64).itemsize
            ),
            "fwht_add_subtracts_per_vector": transform_size * int(math.log2(transform_size)),
        },
        "note": "All optical/electrical parameters are example assumptions, not calibrated experimental values.",
        "optical_path": [
            "white light source",
            "DMD",
            "polarizer",
            "lens1 magnification",
            "object",
            "lens2/lens3 4f shrink system",
            "PEM",
            "orthogonal analyzer",
            "collector lens",
            "voltage-output detector",
            "lock-in amplifier",
            "computer reconstruction",
        ],
        "parameters": {
            "SOURCE_POWER_W": SOURCE_POWER_W,
            "DMD_REFLECTIVITY": DMD_REFLECTIVITY,
            "POLARIZER_TRANSMISSION": POLARIZER_TRANSMISSION,
            "LENS1_MAGNIFICATION": LENS1_MAGNIFICATION,
            "FOUR_F_SHRINK_MAGNIFICATION": FOUR_F_SHRINK_MAGNIFICATION,
            "DMD_PATTERN_DIAMETER_MM": DMD_PATTERN_DIAMETER_MM,
            "object_beam_diameter_mm": object_beam_diameter_mm,
            "pem_input_beam_diameter_mm": pem_input_beam_diameter_mm,
            "PEM_CLEAR_APERTURE_MM": PEM_CLEAR_APERTURE_MM,
            "pem_aperture_ok": pem_aperture_ok,
            "PEM_FREQUENCY_HZ": PEM_FREQUENCY_HZ,
            "PEM_DELTA0_RAD": PEM_DELTA0_RAD,
            "PEM_AXIS_DEG": PEM_AXIS_DEG,
            "DETECTOR_RESPONSIVITY_A_PER_W": DETECTOR_RESPONSIVITY_A_PER_W,
            "TRANSIMPEDANCE_GAIN_V_PER_A": TRANSIMPEDANCE_GAIN_V_PER_A,
            "LOCKIN_COMPONENT_NOISE_STD_V": LOCKIN_COMPONENT_NOISE_STD_V,
            "TIME_SIGNAL_NOISE_STD_V": TIME_SIGNAL_NOISE_STD_V,
        },
        "bessel_coefficients_single_wavelength": {
            "J0": bessel_j(0, PEM_DELTA0_RAD),
            "J1": bessel_j(1, PEM_DELTA0_RAD),
            "J2": bessel_j(2, PEM_DELTA0_RAD),
        },
        "white_light_rgb_simplified_estimate": white_light_info,
        "example_pattern_index": k_ex,
        "example_positive_mask_stokes_projection_W": {
            "S0_plus": s0_plus,
            "S1_plus": s1_plus,
            "S3_plus": s3_plus,
        },
        "example_numerical_lockin_from_V_t": lockin_example,
        "metrics": {
            "transmittance_rmse": rmse,
            "transmittance_mae": mae,
            "transmittance_psnr_db": psnr,
            "transmittance_ssim": ssim,
            "transmittance_max_absolute_error": max_absolute_error,
            "transmittance_pearson_correlation": transmittance_correlation,
            "S0_rmse_W": s0_rmse,
            "S1_rmse_W": s1_rmse,
            "S3_rmse_W": s3_rmse,
            "transmittance": transmittance_metrics,
            "stokes_per_component": stokes_component_metrics,
            "integrated_stokes": {
                "ground_truth": ground_truth_stokes_summary,
                "reconstruction": reconstruction_stokes_summary,
            },
        },
        "limitations": [
            "Current model uses a simplified single-wavelength PEM reconstruction for the main image.",
            "S2 is not reconstructed in this single orthogonal-analyzer configuration.",
            "Lens diffraction, aberration, DMD micromirror diffraction, and real white-light propagation are not fully modeled.",
            "The image vector is zero-padded to a power of two and processed by FWHT; the padded pixels are virtual zero-transmission pixels and are cropped after inversion.",
            "Full sampling still requires one signed measurement per FWHT coefficient, or two physical DMD exposures per complementary pair.",
            "Real experiment requires calibration of PEM axis, delta0(lambda), analyzer angle, detector voltage gain, lock-in phase, and DMD timing.",
        ],
    }

    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    summary = []
    summary.append("基于实际光路的 PEM 偏振单像素成像仿真摘要")
    summary.append("")
    summary.append("实际链路：白光光源 -> DMD -> 起偏器 -> 透镜1放大 -> 物体 -> 4f缩束 -> PEM -> 正交检偏器 -> 聚光透镜 -> 电压输出探测器 -> 锁相 -> 重建")
    summary.append("")
    summary.append("重要说明：本代码中的参数为示例假设，不是实验标定值。")
    summary.append("探测器输出量为电压 V(t)，不是电流。")
    summary.append("")
    summary.append(f"输入图像: {INPUT_IMAGE_PATH}")
    summary.append(f"输出目录: {OUTPUT_DIR}")
    summary.append(f"原始仿真图像尺寸: {image_height} x {image_width}")
    summary.append(f"物理像素数: {n_pixels}")
    summary.append(f"FWHT补零长度: {transform_size}")
    summary.append(f"虚拟补零像素数: {padding_pixels}")
    summary.append(f"有符号Hadamard投影数: {transform_size}")
    summary.append(f"互补DMD物理曝光数: {2 * transform_size}")
    summary.append("")
    summary.append("光束尺寸检查：")
    summary.append(f"  DMD图案等效直径: {DMD_PATTERN_DIAMETER_MM:.3f} mm")
    summary.append(f"  透镜1放大后物体处直径: {object_beam_diameter_mm:.3f} mm")
    summary.append(f"  4f缩束后进入PEM直径: {pem_input_beam_diameter_mm:.3f} mm")
    summary.append(f"  PEM有效口径: {PEM_CLEAR_APERTURE_MM:.3f} mm")
    summary.append(f"  是否匹配PEM口径: {pem_aperture_ok}")
    summary.append("")
    summary.append("相对光功率密度变化：")
    summary.append(f"  透镜1放大导致面积变为 {area_after_lens1_factor:.3f} 倍，功率密度约为原来的 {irradiance_after_lens1_relative:.3f}")
    summary.append(f"  4f缩束导致面积变为物体处的 {area_after_4f_factor:.3f} 倍，功率密度约变为 {irradiance_after_4f_relative_to_object:.3f} 倍")
    summary.append("")
    summary.append("PEM/检偏器简化关系：")
    summary.append("  P_det(t) = eta_post/2 * [S0 - S1*cos(delta(t)) + S3*sin(delta(t))]")
    summary.append("  DC -> S0, f -> S3, 2f -> S1")
    summary.append("  当前单构型不能完整恢复 S2。")
    summary.append("")
    summary.append("重建质量：")
    summary.append(f"  transmittance RMSE: {rmse:.6e}")
    summary.append(f"  transmittance MAE:  {mae:.6e}")
    summary.append(f"  transmittance PSNR: {psnr:.3f} dB")
    summary.append(f"  transmittance SSIM: {ssim:.6f}")
    summary.append(f"  transmittance max absolute error: {max_absolute_error:.6e}")
    summary.append(f"  transmittance Pearson r: {transmittance_correlation:.6f}")
    for component in ["S0", "S1", "S3"]:
        component_result = stokes_component_metrics[component]
        summary.append(
            f"  {component}: RMSE={component_result['rmse_W_per_pixel']:.6e} W/pixel, "
            f"NRMSE={100 * component_result['nrmse_by_gt_peak']:.4f}%, "
            f"Pearson r={component_result['pearson_correlation']:.6f}"
        )
    summary.append("")
    summary.append("整幅图像积分 Stokes 参数：")
    gt_total = ground_truth_stokes_summary["total_stokes_W"]
    rec_total = reconstruction_stokes_summary["total_stokes_W"]
    gt_norm = ground_truth_stokes_summary["normalized_stokes"]
    rec_norm = reconstruction_stokes_summary["normalized_stokes"]
    summary.append(
        "  GT  [S0,S1,S2,S3] / W: "
        f"[{gt_total['S0_W']:.9e}, {gt_total['S1_W']:.9e}, "
        f"{gt_total['S2_W']:.9e}, {gt_total['S3_W']:.9e}]"
    )
    summary.append(
        "  REC [S0,S1,S2,S3] / W: "
        f"[{rec_total['S0_W']:.9e}, {rec_total['S1_W']:.9e}, "
        f"{rec_total['S2_W']:.9e}, {rec_total['S3_W']:.9e}]"
    )
    summary.append(
        "  GT  [s0,s1,s2,s3]: "
        f"[{gt_norm['s0']:.9g}, {gt_norm['s1']:.9g}, "
        f"{gt_norm['s2']:.9g}, {gt_norm['s3']:.9g}]"
    )
    summary.append(
        "  REC [s0,s1,s2,s3]: "
        f"[{rec_norm['s0']:.9g}, {rec_norm['s1']:.9g}, "
        f"{rec_norm['s2']:.9g}, {rec_norm['s3']:.9g}]"
    )
    summary.append("")
    summary.append("主要输出文件：")
    summary.append("  01_preprocessed_input_transmittance.png")
    summary.append("  02_dmd_mask_examples.png")
    summary.append("  03_ground_truth_stokes_physical_units.png")
    summary.append("  04_example_detector_voltage_signal.png")
    summary.append("  05_example_voltage_spectrum.png")
    summary.append("  06_lockin_components_curves.png")
    summary.append("  06_lockin_components.csv")
    summary.append("  07_reconstructed_transmittance.png")
    summary.append("  07_absolute_error_map.png")
    summary.append("  07_transmittance_reconstruction_summary.png")
    summary.append("  08_stokes_reconstruction_contact_sheet.png")
    summary.append("  09_stokes_quantitative_summary.png")
    summary.append("  params.json")

    with open(out_dir / "simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))


if __name__ == "__main__":
    main()
