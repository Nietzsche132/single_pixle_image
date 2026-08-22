import numpy as np
from PIL import Image
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import subprocess  # <-- 修正2: 添加缺失的导入

# ==============================================================================
#                                 配置区域
# ==============================================================================

# --- 主要配置 ---
OUTPUT_DIR = Path(r"F:\fxfy[-64,63]")
FX_RANGE = (-64, 63)
FY_RANGE = (0, 63)
PHASES = [0, 90, 180, 270]

# --- 图像参数 ---
BASE_SIZE = 128
SCALE_FACTOR = 8
TARGET_SIZE = (1920, 1080)

# --- 性能配置 ---
SKIP_EXISTING = True
USE_MULTIPROCESSING = True
MAX_WORKERS = os.cpu_count()


class FourierBasisGenerator:
    """
    高效的傅里叶基底图案生成器
    """

    def __init__(self, base_size: int = 128, scale_factor: int = 8, target_size: Tuple[int, int] = (1920, 1080)):
        self.base_size = base_size
        self.scale_factor = scale_factor
        self.target_width, self.target_height = target_size
        self.enlarged_size = (base_size * scale_factor, base_size * scale_factor)
        x = np.arange(self.base_size)
        y = np.arange(self.base_size)
        self.X, self.Y = np.meshgrid(x, y)

    def generate_basis(self, fx: int, fy: int, phase_deg: int) -> Image.Image:
        phase_rad = np.deg2rad(phase_deg)
        cosine_wave = 128 + 127 * np.cos(
            2 * np.pi * (fx * self.X / self.base_size + fy * self.Y / self.base_size) + phase_rad
        )
        cosine_wave = np.clip(cosine_wave, 0, 255).astype(np.uint8)
        return Image.fromarray(cosine_wave, mode='L')

    def enlarge_image(self, image: Image.Image) -> Image.Image:
        return image.resize(self.enlarged_size, Image.BICUBIC)

    def dither_image(self, image: Image.Image) -> Image.Image:
        return image.convert('1', dither=Image.FLOYDSTEINBERG)

    def pad_to_target(self, image: Image.Image) -> Image.Image:
        padded = Image.new('1', (self.target_width, self.target_height), color=0)
        img_width, img_height = image.size
        x_offset = (self.target_width - img_width) // 2
        y_offset = (self.target_height - img_height) // 2
        padded.paste(image, (x_offset, y_offset))
        return padded

    def process_single_basis(self, idx: int, fx: int, fy: int, phase: int, output_dir: Path, skip_existing: bool) -> \
    Tuple[bool, Path, str]:
        filename = f"{idx:05d}_fx={fx}_fy={fy}_phase={phase}.bmp"
        filepath = output_dir / filename
        if skip_existing and filepath.exists():
            return True, filepath, "已存在，跳过"
        try:
            basis = self.generate_basis(fx, fy, phase)
            enlarged = self.enlarge_image(basis)
            dithered = self.dither_image(enlarged)
            final_image = self.pad_to_target(dithered)
            final_image.save(filepath, format='BMP')
            return True, filepath, "成功"
        except Exception as e:
            return False, filepath, str(e)

    def generate_all_bases(self, fx_range: Tuple[int, int], fy_range: Tuple[int, int], phases: List[int],
                           output_dir: Path, skip_existing: bool, use_multiprocessing: bool, max_workers: int | None):
        output_dir.mkdir(parents=True, exist_ok=True)
        param_combinations = []
        fx_values = range(fx_range[0], fx_range[1] + 1)
        fy_values = range(fy_range[0], fy_range[1] + 1)
        for fx in fx_values:
            for fy in fy_values:
                if fx == 0 and fy == 0:
                    param_combinations.append((fx, fy, 0))
                else:
                    for phase in phases:
                        param_combinations.append((fx, fy, phase))
        tasks_with_index = list(enumerate(param_combinations, 1))
        total = len(tasks_with_index)

        print("=" * 60)
        print("开始批量生成傅里叶基底图案 (已优化版本)")
        print(f"频率范围: fx={list(fx_range)}, fy={list(fy_range)}")
        print(f"相位: {phases}")
        print(f"总任务数: {total}")
        print(f"输出目录: {output_dir}")
        print(f"使用多进程: {'是' if use_multiprocessing else '否'} (最大进程数: {max_workers})")
        print("=" * 60)

        start_time = time.time()
        results = []
        if use_multiprocessing and total > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_basis, idx, fx, fy, phase, output_dir, skip_existing): (fx, fy,
                                                                                                                phase)
                    for idx, (fx, fy, phase) in tasks_with_index
                }
                with tqdm(total=total, desc="生成进度") as pbar:
                    for future in as_completed(futures):
                        fx, fy, phase = futures[future]
                        try:
                            success, filepath, message = future.result()
                            results.append(
                                {'success': success, 'filepath': filepath, 'message': message, 'fx': fx, 'fy': fy,
                                 'phase': phase})
                        except Exception as e:
                            results.append({'success': False, 'filepath': None, 'message': str(e), 'fx': fx, 'fy': fy,
                                            'phase': phase})
                        pbar.update(1)
        else:
            for idx, (fx, fy, phase) in tqdm(tasks_with_index, desc="生成进度"):
                success, filepath, message = self.process_single_basis(idx, fx, fy, phase, output_dir, skip_existing)
                results.append(
                    {'success': success, 'filepath': filepath, 'message': message, 'fx': fx, 'fy': fy, 'phase': phase})

        end_time = time.time()
        # <-- 修正1: 在调用print_and_save_report时，传递skip_existing参数
        self.print_and_save_report(results, output_dir, fx_range, fy_range, phases, start_time, end_time, skip_existing)

    # <-- 修正1: 在函数定义中接收skip_existing参数
    def print_and_save_report(self, results: List[Dict], output_dir: Path, fx_range: Tuple[int, int],
                              fy_range: Tuple[int, int], phases: List[int], start_time: float, end_time: float,
                              skip_existing: bool):
        """打印并保存生成报告"""
        elapsed_time = end_time - start_time
        total = len(results)
        successful = sum(1 for r in results if r['success'] and r['message'] == "成功")
        skipped = sum(1 for r in results if r['message'] == "已存在，跳过")
        failed = total - successful - skipped

        print("=" * 60)
        print("生成完成!")
        print(f"总任务数: {total}")
        print(f"成功生成: {successful}")
        print(f"跳过已存在: {skipped}")
        print(f"失败: {failed}")
        print(f"总耗时: {elapsed_time:.2f} 秒")
        if total > 0:
            print(f"平均每个新图案耗时: {elapsed_time / successful if successful > 0 else 0:.4f} 秒")

        print("-" * 60)
        print("生成纯黑参考图案...")
        black_filename = "00000_black.bmp"
        black_filepath = output_dir / black_filename

        # <-- 修正1: 现在这里的 'skip_existing' 是一个已定义的变量，可以正常使用
        if skip_existing and black_filepath.exists():
            print(f"纯黑图案已存在，跳过: {black_filepath}")
        else:
            try:
                black_img = Image.new('1', TARGET_SIZE, color=0)
                black_img.save(black_filepath, format='BMP')
                print(f"成功生成纯黑图案: {black_filepath}")
            except Exception as e:
                print(f"生成纯黑图案失败: {e}")
                failed += 1

        if failed > 0:
            print("\n失败的任务详情:")
            for r in results:
                if not r['success'] and r['message'] != "已存在，跳过":
                    print(f"  fx={r['fx']}, fy={r['fy']}, phase={r['phase']}: {r['message']}")

        report = {
            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'base_size': self.base_size, 'scale_factor': self.scale_factor,
                'target_size': f"{self.target_width}x{self.target_height}",
                'fx_range': list(fx_range), 'fy_range': list(fy_range), 'phases': phases
            },
            'statistics': {
                'total_tasks': total, 'successful': successful, 'skipped': skipped,
                'failed': failed, 'elapsed_time_sec': round(elapsed_time, 2)
            },
            'file_list': [
                {'filename': r['filepath'].name, 'fx': r['fx'], 'fy': r['fy'], 'phase': r['phase'],
                 'status': r['message']}
                for r in results if r['filepath']
            ]
        }
        json_path = output_dir / "generation_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print(f"生成报告已保存到: {json_path}")
        print("所有任务执行完毕。")


def main():
    """主程序入口"""
    try:
        from tqdm import tqdm
    except ImportError:
        print("检测到tqdm未安装，正在尝试自动安装...")
        try:
            # <-- 修正2: 这里的 'subprocess' 现在可以被正确解析
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
            print("tqdm安装成功！")
        except Exception as e:
            print(f"自动安装tqdm失败: {e}")
            print("请手动运行 'pip install tqdm' 后再执行脚本。")
            return

    generator = FourierBasisGenerator(
        base_size=BASE_SIZE,
        scale_factor=SCALE_FACTOR,
        target_size=TARGET_SIZE
    )

    generator.generate_all_bases(
        fx_range=FX_RANGE,
        fy_range=FY_RANGE,
        phases=PHASES,
        output_dir=OUTPUT_DIR,
        skip_existing=SKIP_EXISTING,
        use_multiprocessing=USE_MULTIPROCESSING,
        max_workers=MAX_WORKERS
    )


if __name__ == "__main__":
    main()