"""
基于实际光路的 PEM 偏振单像素成像仿真

实际光路：
白光光源 -> DMD -> 起偏器 -> 透镜1放大 -> 物体 -> 透镜2/透镜3的4f缩束系统 -> PEM -> 与起偏器垂直的检偏器
 -> 聚光透镜 -> 探测器输出电压-> 锁相放大器提取 DC/f/2f -> 计算机算法重建。

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

# 示例假设：仿真重建分辨率。必须满足 IMAGE_SIZE**2 是 2 的整数次幂。
# 32x32 运算很快；可改成 64x64，但图案数会变成 4096。
IMAGE_SIZE = 32

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


def hadamard_matrix(n: int) -> np.ndarray:
    """生成 n x n Hadamard 矩阵，n 必须是 2 的整数次幂。"""
    if not is_power_of_two(n):
        raise ValueError(f"Hadamard size n={n} is not a power of two.")
    h = np.array([[1]], dtype=np.int8)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]]).astype(np.int8)
    return h


def resize_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return img.resize(size, resample)


def load_input_image(path: str, image_size: int, invert: bool = False) -> np.ndarray:
    """
    读取输入图片，转为灰度并归一化到 [0,1]。
    这里将灰度值解释为物体强度透过率：白色=高透过，黑色=低透过。
    """
    img = Image.open(path).convert("L")
    img = resize_image(img, (image_size, image_size))
    arr = np.asarray(img, dtype=np.float64) / 255.0
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
    y_all = np.concatenate(arrays)
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
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
        pts = []
        for j, val in enumerate(y):
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

    # 假设 DMD 全开时，功率均匀分布到 IMAGE_SIZE x IMAGE_SIZE 个像素。
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

def save_dmd_mask_examples(h: np.ndarray, out_dir: Path, image_size: int) -> None:
    """保存若干 DMD 互补正图案 M+ 的示例。"""
    imgs = []
    for k in range(min(NUM_MASK_EXAMPLES, h.shape[0])):
        mask_plus = ((h[k] + 1) / 2).reshape(image_size, image_size)
        imgs.append(arr_to_labeled_image(mask_plus, f"DMD M+ #{k}", size_px=160))
    sheet = make_contact_sheet(imgs, cols=4)
    sheet.save(out_dir / "02_dmd_mask_examples.png")


def save_stokes_contact_sheet(
    s0_gt: np.ndarray, s1_gt: np.ndarray, s3_gt: np.ndarray,
    s0_rec: np.ndarray, s1_rec: np.ndarray, s3_rec: np.ndarray,
    out_dir: Path,
) -> None:
    imgs = [
        arr_to_labeled_image(s0_gt, "GT S0", normalize=True),
        arr_to_labeled_image(s1_gt, "GT S1", normalize=True),
        arr_to_labeled_image(s3_gt, "GT S3", normalize=True),
        arr_to_labeled_image(s0_rec, "REC S0", normalize=True),
        arr_to_labeled_image(s1_rec, "REC S1", normalize=True),
        arr_to_labeled_image(s3_rec, "REC S3", normalize=True),
    ]
    sheet = make_contact_sheet(imgs, cols=3)
    sheet.save(out_dir / "08_stokes_reconstruction_contact_sheet.png")


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

    n_pixels = IMAGE_SIZE * IMAGE_SIZE
    if not is_power_of_two(n_pixels):
        raise ValueError("IMAGE_SIZE**2 must be a power of two for Hadamard coding.")

    # --------------------------------------------------------
    # 1) 读取输入图像，作为待成像物体强度透过率
    # --------------------------------------------------------
    transmittance = load_input_image(INPUT_IMAGE_PATH, IMAGE_SIZE, INVERT_INPUT_IMAGE)
    save_gray_image(transmittance, out_dir / "01_preprocessed_input_transmittance.png")

    # --------------------------------------------------------
    # 2) 构造物体后的 Stokes 分布：S0, S1, S3
    # --------------------------------------------------------
    s0_map, s1_map, _s2_map, s3_map = build_object_stokes(transmittance)

    save_gray_image(s0_map / np.max(s0_map + 1e-30), out_dir / "03_ground_truth_S0_normalized.png")
    save_gray_image(np.abs(s1_map) / np.max(np.abs(s1_map) + 1e-30), out_dir / "03_ground_truth_abs_S1_normalized.png")
    save_gray_image(np.abs(s3_map) / np.max(np.abs(s3_map) + 1e-30), out_dir / "03_ground_truth_abs_S3_normalized.png")

    # --------------------------------------------------------
    # 3) 生成 DMD Hadamard 编码
    # --------------------------------------------------------
    h = hadamard_matrix(n_pixels)
    save_dmd_mask_examples(h, out_dir, IMAGE_SIZE)

    s0_flat = s0_map.reshape(-1)
    s1_flat = s1_map.reshape(-1)
    s3_flat = s3_map.reshape(-1)

    # Hadamard 投影：等效于 DMD 互补图案 M+ - M- 的测量差分
    s0_proj = h.astype(np.float64) @ s0_flat
    s1_proj = h.astype(np.float64) @ s1_flat
    s3_proj = h.astype(np.float64) @ s3_flat

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
    # H^{-1} = H^T / N
    h_float = h.astype(np.float64)
    s0_rec = (h_float.T @ s0_hat_proj / n_pixels).reshape(IMAGE_SIZE, IMAGE_SIZE)
    s1_rec = (h_float.T @ s1_hat_proj / n_pixels).reshape(IMAGE_SIZE, IMAGE_SIZE)
    s3_rec = (h_float.T @ s3_hat_proj / n_pixels).reshape(IMAGE_SIZE, IMAGE_SIZE)

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

    save_gray_image(transmittance_rec, out_dir / "07_reconstructed_transmittance.png")
    save_gray_image(np.abs(error_map), out_dir / "07_absolute_error_map.png", normalize=True)
    save_stokes_contact_sheet(s0_map, s1_map, s3_map, s0_rec, s1_rec, s3_rec, out_dir)

    # --------------------------------------------------------
    # 7) 生成一个实际物理 DMD 正图案 M+ 下的探测器电压 V(t)
    # --------------------------------------------------------
    k_ex = max(0, min(EXAMPLE_PATTERN_INDEX, n_pixels - 1))
    mask_plus = ((h[k_ex] + 1) / 2).astype(np.float64)

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
    rmse = float(np.sqrt(np.mean((transmittance_rec - transmittance) ** 2)))
    mae = float(np.mean(np.abs(transmittance_rec - transmittance)))
    peak = 1.0
    psnr = float(20 * math.log10(peak / (rmse + 1e-30)))

    s0_rmse = float(np.sqrt(np.mean((s0_rec - s0_map) ** 2)))
    s1_rmse = float(np.sqrt(np.mean((s1_rec - s1_map) ** 2)))
    s3_rmse = float(np.sqrt(np.mean((s3_rec - s3_map) ** 2)))

    # --------------------------------------------------------
    # 10) 保存参数和说明文件
    # --------------------------------------------------------
    white_light_info = white_light_rgb_effective_coefficients()

    params = {
        "input_image_path": INPUT_IMAGE_PATH,
        "output_dir": OUTPUT_DIR,
        "image_size": IMAGE_SIZE,
        "num_pixels": n_pixels,
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
            "S0_rmse_W": s0_rmse,
            "S1_rmse_W": s1_rmse,
            "S3_rmse_W": s3_rmse,
        },
        "limitations": [
            "Current model uses a simplified single-wavelength PEM reconstruction for the main image.",
            "S2 is not reconstructed in this single orthogonal-analyzer configuration.",
            "Lens diffraction, aberration, DMD micromirror diffraction, and real white-light propagation are not fully modeled.",
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
    summary.append(f"仿真图像尺寸: {IMAGE_SIZE} x {IMAGE_SIZE}")
    summary.append(f"DMD Hadamard 图案数: {n_pixels}")
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
    summary.append(f"  S0 RMSE: {s0_rmse:.6e} W")
    summary.append(f"  S1 RMSE: {s1_rmse:.6e} W")
    summary.append(f"  S3 RMSE: {s3_rmse:.6e} W")
    summary.append("")
    summary.append("主要输出文件：")
    summary.append("  01_preprocessed_input_transmittance.png")
    summary.append("  02_dmd_mask_examples.png")
    summary.append("  04_example_detector_voltage_signal.png")
    summary.append("  05_example_voltage_spectrum.png")
    summary.append("  06_lockin_components_curves.png")
    summary.append("  06_lockin_components.csv")
    summary.append("  07_reconstructed_transmittance.png")
    summary.append("  07_absolute_error_map.png")
    summary.append("  08_stokes_reconstruction_contact_sheet.png")
    summary.append("  params.json")

    with open(out_dir / "simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))


if __name__ == "__main__":
    main()
