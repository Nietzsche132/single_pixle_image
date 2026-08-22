from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim


# ==================== Configuration ====================
DATA_NUMBER = 3
DATA_DIR = Path(r"D:\桌面\毕设\data[-64,63]")
ORIGINAL_IMAGE_PATH = Path(r"D:\桌面\毕设\picture\测试图案\number2_128x128.png")

FX_MIN = -64
FX_MAX = 63
FY_MIN = -45
FY_MAX = 44
IMAGE_SIZE = 128
RAW_GROUP_SIZE = 10
TRIM_COUNT = 2

INPUT_FILE = DATA_DIR / f"{DATA_NUMBER}.tdms"
OUTPUT_EXCEL_FILE = DATA_DIR / f"数据{DATA_NUMBER}处理结果.xlsx"
OUTPUT_IMAGE_PATH = DATA_DIR / f"数据{DATA_NUMBER}处理重构.png"
OUTPUT_FFT_PATH = DATA_DIR / f"数据{DATA_NUMBER}处理傅里叶系数幅度.png"
OUTPUT_COMPARISON_PATH = DATA_DIR / f"数据{DATA_NUMBER}处理对比.png"
# =======================================================


def read_measurement_data(filepath):
    """Read measurement data from TDMS, plain text, or CSV."""
    filepath = Path(filepath)
    print(f"Reading measurement file: {filepath}")

    try:
        from nptdms import TdmsFile

        with TdmsFile.open(filepath) as tdms_file:
            groups = tdms_file.groups()
            if not groups:
                raise ValueError("No group found in TDMS file.")

            channels = groups[0].channels()
            if not channels:
                raise ValueError("No channel found in the first TDMS group.")

            data = channels[0][:].tolist()
            print(f"TDMS loaded successfully. Raw points: {len(data)}")
            return data
    except ImportError:
        print("nptdms is not installed. Trying text/CSV fallback...")
    except Exception as exc:
        print(f"TDMS loading failed: {exc}. Trying text/CSV fallback...")

    try:
        data = np.loadtxt(filepath).tolist()
        if np.isscalar(data):
            data = [float(data)]
        print(f"Text file loaded successfully. Raw points: {len(data)}")
        return data
    except Exception as exc:
        print(f"Text loading failed: {exc}. Trying CSV fallback...")

    try:
        data = pd.read_csv(filepath, header=None).iloc[:, 0].tolist()
        print(f"CSV loaded successfully. Raw points: {len(data)}")
        return data
    except Exception as exc:
        raise IOError(f"Unable to read measurement file: {filepath}") from exc


def trimmed_group_averages(data, group_size=10, trim_count=2, start_index=1):
    """Average every group after trimming high/low outliers."""
    averages = []

    for i in range(start_index, len(data), group_size):
        group = sorted(data[i : i + group_size])
        if len(group) >= trim_count * 2 + 2:
            group = group[trim_count:-trim_count]

        if group:
            averages.append(float(np.mean(group)))

    return averages


def generate_frequency_pairs():
    """Generate measured Fourier coordinates in acquisition order."""
    pairs = []
    for fx in range(FX_MIN, FX_MAX + 1):
        for fy in range(FY_MIN, FY_MAX + 1):
            pairs.append((fx, fy))
    return pairs


def build_result_dataframe(averages, frequency_pairs):
    """Build the Excel table and calculate normalized Fourier coefficients."""
    rows = []

    for i, (fx, fy) in enumerate(frequency_pairs):
        values = averages[i * 4 : i * 4 + 4]
        values = values + [np.nan] * (4 - len(values))
        rows.append(
            {
                "fx": fx,
                "fy": fy,
                "fx/fy": f"{fx}, {fy}",
                "D1 (0deg)": values[0],
                "D2 (90deg)": values[1],
                "D3 (180deg)": values[2],
                "D4 (270deg)": values[3],
            }
        )

    result_df = pd.DataFrame(rows)
    result_df["D1-D3"] = result_df["D1 (0deg)"] - result_df["D3 (180deg)"]
    result_df["D2-D4"] = result_df["D2 (90deg)"] - result_df["D4 (270deg)"]

    dc_rows = result_df[(result_df["fx"] == 0) & (result_df["fy"] == 0)]
    normalization_factor = 1.0

    if not dc_rows.empty:
        dc = dc_rows.iloc[0]
        if dc[["D1 (0deg)", "D2 (90deg)", "D3 (180deg)", "D4 (270deg)"]].notna().all():
            first_pair_mean = (dc["D1 (0deg)"] + dc["D2 (90deg)"]) / 2
            second_pair_mean = (dc["D3 (180deg)"] + dc["D4 (270deg)"]) / 2
            normalization_factor = first_pair_mean - second_pair_mean

    if normalization_factor == 0 or pd.isna(normalization_factor):
        print("Warning: invalid normalization factor. Using 1.0 instead.")
        normalization_factor = 1.0

    print(f"Normalization factor: {normalization_factor}")

    result_df["D1-D3 normalized"] = result_df["D1-D3"] / normalization_factor
    result_df["D2-D4 normalized"] = result_df["D2-D4"] / normalization_factor

    complex_coeffs = []
    coeff_text = []

    for _, row in result_df.iterrows():
        real_part = row["D1-D3 normalized"]
        imag_measure = row["D2-D4 normalized"]

        if pd.isna(real_part) or pd.isna(imag_measure):
            complex_coeffs.append(np.nan)
            coeff_text.append("")
            continue

        if row["fx"] == 0 and row["fy"] == 0:
            coeff = complex(float(real_part), 0.0)
        else:
            coeff = complex(float(real_part), -float(imag_measure))

        complex_coeffs.append(coeff)
        coeff_text.append(format_complex(coeff))

    result_df["Fourier coefficient"] = coeff_text
    result_df["_complex_coeff"] = complex_coeffs
    return result_df, normalization_factor


def format_complex(value):
    """Format a complex number for Excel display."""
    if abs(value.imag) < 1e-12:
        return f"{value.real:.5f}"

    sign = "+" if value.imag >= 0 else "-"
    return f"{value.real:.5f} {sign} {abs(value.imag):.5f}j"


def build_full_frequency_matrix(result_df, image_size=128):
    """Fill a 128x128 Fourier matrix with measured and conjugate coefficients."""
    fourier_matrix = np.zeros((image_size, image_size), dtype=complex)

    for _, row in result_df.iterrows():
        coeff = row["_complex_coeff"]
        if not isinstance(coeff, complex):
            continue

        fx = int(row["fx"])
        fy = int(row["fy"])
        set_frequency_value(fourier_matrix, fx, fy, coeff)

        if (fx, fy) != (0, 0):
            set_frequency_value(fourier_matrix, -fx, -fy, np.conj(coeff))

    return fourier_matrix


def set_frequency_value(matrix, fx, fy, value):
    """Set Fourier value using standard numpy FFT indexing."""
    image_size = matrix.shape[0]
    x_index = fx % image_size
    y_index = fy % image_size

    matrix[y_index, x_index] = value


def reconstruct_image(fourier_matrix):
    """Reconstruct and normalize image from the Fourier matrix."""
    reconstructed = np.fft.ifft2(fourier_matrix)
    reconstructed = np.real(reconstructed)
    reconstructed = np.fliplr(reconstructed)

    min_val = np.min(reconstructed)
    max_val = np.max(reconstructed)

    if max_val > min_val:
        normalized = (reconstructed - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(reconstructed)

    return normalized.astype(np.uint8)


def save_fft_amplitude_image(fourier_matrix, output_path):
    """Save a viewable log-amplitude spectrum image."""
    amplitude = np.log1p(np.abs(np.fft.fftshift(fourier_matrix)))
    min_val = np.min(amplitude)
    max_val = np.max(amplitude)

    if max_val > min_val:
        amplitude = (amplitude - min_val) / (max_val - min_val) * 255
    else:
        amplitude = np.zeros_like(amplitude)

    Image.fromarray(amplitude.astype(np.uint8), mode="L").save(output_path)
    print(f"Fourier amplitude image saved to: {output_path}")


def evaluate_and_save_comparison(reconstructed_image):
    """Calculate MSE/PSNR/SSIM and save comparison image if original exists."""
    try:
        original_img = Image.open(ORIGINAL_IMAGE_PATH).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception as exc:
        print(f"Original image could not be loaded, skipping comparison: {exc}")
        return

    original_array = np.array(original_img)

    # Keep this calculation consistent with data[-32,32]数据处理.py.
    # np.mean already divides by the total number of pixels.
    mse = np.mean((original_array - reconstructed_image) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    ssim_value = ssim(original_array, reconstructed_image, data_range=255)

    # This is the mathematically strict MSE. It is printed only as a check,
    # because uint8 subtraction in the legacy script can wrap around.
    mse_float_check = np.mean(
        (original_array.astype(np.float64) - reconstructed_image.astype(np.float64)) ** 2
    )

    print("Image quality:")
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE float-check: {mse_float_check:.2f}")

    margin = 30
    gap = 20
    title_height = 40
    info_height = 45
    width = IMAGE_SIZE * 2 + gap + margin * 2
    height = IMAGE_SIZE + title_height + info_height + margin
    comparison = Image.new("RGB", (width, height), color=(255, 255, 255))

    comparison.paste(original_img, (margin, title_height))
    comparison.paste(Image.fromarray(reconstructed_image, mode="L"), (margin + IMAGE_SIZE + gap, title_height))

    draw = ImageDraw.Draw(comparison)
    title_font, info_font = load_fonts()

    draw.text((margin + IMAGE_SIZE // 2, 18), "Original", fill=(0, 0, 0), font=title_font, anchor="mm")
    draw.text(
        (margin + IMAGE_SIZE + gap + IMAGE_SIZE // 2, 18),
        "重构图像",
        fill=(0, 0, 0),
        font=title_font,
        anchor="mm",
    )

    info_text = f"MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}"
    draw.text((margin, title_height + IMAGE_SIZE + 14), info_text, fill=(0, 0, 0), font=info_font)

    comparison.save(OUTPUT_COMPARISON_PATH, quality=95)
    print(f"Comparison image saved to: {OUTPUT_COMPARISON_PATH}")


def load_fonts():
    """Load common Windows fonts, falling back to PIL default."""
    for font_name in ("simsun.ttc", "simhei.ttf", "msyh.ttc"):
        try:
            return ImageFont.truetype(font_name, 18), ImageFont.truetype(font_name, 14)
        except OSError:
            continue

    default_font = ImageFont.load_default()
    return default_font, default_font


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = read_measurement_data(INPUT_FILE)
    averages = trimmed_group_averages(
        raw_data,
        group_size=RAW_GROUP_SIZE,
        trim_count=TRIM_COUNT,
        start_index=1,
    )
    print(f"Averaged values: {len(averages)}")

    frequency_pairs = generate_frequency_pairs()
    required_average_count = len(frequency_pairs) * 4
    print(f"Fourier points: {len(frequency_pairs)}")
    print(f"Required averaged values: {required_average_count}")

    if len(averages) < required_average_count:
        print(
            "Warning: averaged values are fewer than required. "
            "Missing D1/D2/D3/D4 values will be left blank in Excel."
        )

    result_df, _ = build_result_dataframe(averages, frequency_pairs)

    excel_df = result_df.drop(columns=["_complex_coeff"])
    excel_df.to_excel(OUTPUT_EXCEL_FILE, index=False)
    print(f"Excel file saved to: {OUTPUT_EXCEL_FILE}")

    fourier_matrix = build_full_frequency_matrix(result_df, image_size=IMAGE_SIZE)
    reconstructed_image = reconstruct_image(fourier_matrix)

    Image.fromarray(reconstructed_image, mode="L").save(OUTPUT_IMAGE_PATH)
    print(f"Reconstructed image saved to: {OUTPUT_IMAGE_PATH}")

    save_fft_amplitude_image(fourier_matrix, OUTPUT_FFT_PATH)
    evaluate_and_save_comparison(reconstructed_image)

    print("All processing completed.")


if __name__ == "__main__":
    main()
