import os
import shutil
import random
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ==============================================================================
#                                 配置区域
# ==============================================================================

# 1. 路径配置
SOURCE_DIR = Path(r"D:\桌面\毕设\fxfy[-64,63]")
DEST_DIR = Path(r"D:\桌面\毕设\压缩感知\40%_1")

# 2. 采样率配置
TARGET_SAMPLING_RATE = 0.40  # 40%

# 3. 频率范围配置
FX_TOTAL_RANGE = (-64, 63)
FY_TOTAL_RANGE = (0, 63)
FX_CORE_RANGE = (-25, 25)
FY_CORE_RANGE = (0, 25)


# ==============================================================================
#                                 主逻辑
# ==============================================================================

def main():
    """
    主执行函数
    """
    # --- 步骤 1: 初始化 ---
    print("--- 步骤 1: 初始化和安全检查 ---")
    if not SOURCE_DIR.exists():
        print(f"错误: 源文件夹不存在 -> {SOURCE_DIR}")
        return
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"源文件夹: {SOURCE_DIR}")
    print(f"目标文件夹: {DEST_DIR}")
    print("-" * 30)

    # --- 步骤 2: 计算采样频点 ---
    print("--- 步骤 2: 计算采样频点 ---")
    all_freq_points = set()
    for fx in range(FX_TOTAL_RANGE[0], FX_TOTAL_RANGE[1] + 1):
        for fy in range(FY_TOTAL_RANGE[0], FY_TOTAL_RANGE[1] + 1):
            all_freq_points.add((fx, fy))

    total_freq_points_count = len(all_freq_points)
    print(f"总频点数量: {total_freq_points_count}")

    core_freq_points = set()
    for fx in range(FX_CORE_RANGE[0], FX_CORE_RANGE[1] + 1):
        for fy in range(FY_CORE_RANGE[0], FY_CORE_RANGE[1] + 1):
            if (fx, fy) in all_freq_points:
                core_freq_points.add((fx, fy))

    randomizable_freq_points = list(all_freq_points - core_freq_points)

    core_points_count = len(core_freq_points)
    print(f"中心密集采样区频点数 (必选): {core_points_count}")

    total_points_to_sample = int(total_freq_points_count * TARGET_SAMPLING_RATE)
    random_points_to_sample_count = total_points_to_sample - core_points_count

    if random_points_to_sample_count < 0:
        print(f"警告: 中心区频点数({core_points_count})已超过目标采样数({total_points_to_sample})。")
        random_points_to_sample_count = 0
    elif random_points_to_sample_count > len(randomizable_freq_points):
        print(f"警告: 计算出的随机采样数({random_points_to_sample_count})超过了可用的外围频点数。")
        random_points_to_sample_count = len(randomizable_freq_points)

    print(f"目标总采样频点数 ({TARGET_SAMPLING_RATE:.0%}): {total_points_to_sample}")
    print(f"需要从外围随机采样的频点数: {random_points_to_sample_count}")

    randomly_selected_points = random.sample(randomizable_freq_points, random_points_to_sample_count)

    final_selected_points = core_freq_points.union(set(randomly_selected_points))

    final_selected_count = len(final_selected_points)
    actual_sampling_rate = final_selected_count / total_freq_points_count

    print(f"最终确定的采样频点总数: {final_selected_count}")
    print(f"实际采样率: {actual_sampling_rate:.2%}")
    print("-" * 30)

    # --- 步骤 3: 复制文件 (与之前相同) ---
    print("--- 步骤 3: 开始复制文件 ---")
    pattern = re.compile(r"_fx=(-?\d+)_fy=(\d+)")
    source_files = list(SOURCE_DIR.glob('*.bmp'))
    copied_files_count = 0

    for file_path in tqdm(source_files, desc="复制文件进度"):
        match = pattern.search(file_path.name)
        if match:
            fx, fy = int(match.group(1)), int(match.group(2))
            if (fx, fy) in final_selected_points:
                shutil.copy2(file_path, DEST_DIR)
                copied_files_count += 1
    print("文件复制完成。")
    print("-" * 30)

    # --- 新增步骤 4: 生成并保存采样矩阵 ---
    print("--- 步骤 4: 生成并保存采样矩阵 (Mask) ---")

    # 定义矩阵维度
    fx_dim = FX_TOTAL_RANGE[1] - FX_TOTAL_RANGE[0] + 1  # 63 - (-64) + 1 = 128
    fy_dim = FY_TOTAL_RANGE[1] - FY_TOTAL_RANGE[0] + 1  # 63 - 0 + 1 = 64

    # 初始化一个全为0的矩阵
    sampling_matrix = np.zeros((fy_dim, fx_dim), dtype=np.uint8)

    # 定义fx到数组列索引的映射函数
    fx_offset = -FX_TOTAL_RANGE[0]  # fx_offset = 64

    # 遍历所有选中的频点，在矩阵的对应位置标为1
    for fx, fy in final_selected_points:
        col_idx = fx + fx_offset  # 映射 fx:[-64, 63] -> col_idx:[0, 127]
        row_idx = fy  # fy: [0, 63] -> row_idx:[0, 63]

        # 边界检查 (虽然理论上不会超出，但好习惯)
        if 0 <= row_idx < fy_dim and 0 <= col_idx < fx_dim:
            sampling_matrix[row_idx, col_idx] = 1

    # 定义保存路径 (保存在目标文件夹内)
    matrix_basename = "sampling_matrix_40p"
    npy_path = DEST_DIR / f"{matrix_basename}.npy"
    csv_path = DEST_DIR / f"{matrix_basename}.csv"
    img_path = DEST_DIR / f"{matrix_basename}.png"

    # 1. 保存为 .npy 文件 (推荐)
    np.save(npy_path, sampling_matrix)
    print(f"采样矩阵已保存为Numpy格式: {npy_path}")

    # 2. 保存为 .csv 文件 (方便查看)
    np.savetxt(csv_path, sampling_matrix, fmt='%d', delimiter=',')
    print(f"采样矩阵已保存为CSV格式: {csv_path}")

    # 3. 保存为可视化图片 .png
    # 将 0 -> 0 (黑色), 1 -> 255 (白色)
    matrix_image_data = sampling_matrix * 255
    matrix_image = Image.fromarray(matrix_image_data, 'L')
    matrix_image.save(img_path)
    print(f"采样矩阵可视化图片已保存: {img_path}")
    print("-" * 30)

    # --- 步骤 5: 最终报告 (更新) ---
    print("--- 最终报告 ---")
    print(f"处理完成！")
    print(f"总频点数: {total_freq_points_count}")
    print(f"目标采样率: {TARGET_SAMPLING_RATE:.2%}")
    print(f"采样频点详情:")
    print(f"  - 中心区频点 (全部采样): {core_points_count}")
    print(f"  - 外围区频点 (随机采样): {random_points_to_sample_count}")
    print(f"  - 总计采样频点: {final_selected_count} (实际采样率: {actual_sampling_rate:.2%})")
    print(f"共复制了 {copied_files_count} 个图片文件到目标文件夹。")
    print(f"采样矩阵 (Mask) 已生成并保存在目标文件夹中。")
    print(f"-> {DEST_DIR}")
    print("=" * 30)


if __name__ == "__main__":
    # 确保依赖已安装
    try:
        import numpy
        from PIL import Image
    except ImportError as e:
        print(f"错误: 缺少必要的库 -> {e.name}")
        print(f"请运行 'pip install numpy Pillow' 来安装。")
    else:
        main()