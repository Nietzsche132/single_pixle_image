"""
生成单一频率的傅里叶基底图像测试，修改频率在第11行的fx和fy，修改保存目录在第90行
代码运行的逻辑为：
1.生成128×128的余弦波基底 → 2. 放大到1024×1024 → 3. 抖动二值化 → 4. 填充到1920×1080 → 5. 保存为BMP

"""
import numpy as np
from PIL import Image
import os

def generate_fourier_basis(size=128, fx=3, fy=3, phase_deg=0):
    """
    定义基底图片大小 和 频率 ，生成二维傅里叶基底（8bit灰度）
    """
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    phase_rad = np.deg2rad(phase_deg)
    # 生成余弦波，范围0-255
    cosine_wave = 128 + 127 * np.cos(2 * np.pi * (fx * X / size + fy * Y / size) + phase_rad)
    cosine_wave = np.clip(cosine_wave, 0, 255).astype(np.uint8)

    return Image.fromarray(cosine_wave, mode='L')


def floyd_steinberg_dithering(image):
    """
    Floyd-Steinberg抖动法二值化
    """
    img_array = np.array(image, dtype=np.float32)
    height, width = img_array.shape

    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = img_array[y, x]
            new_pixel = 255 if old_pixel >= 128 else 0
            img_array[y, x] = new_pixel

            error = old_pixel - new_pixel

            img_array[y, x + 1] += error * 7 / 16
            img_array[y + 1, x - 1] += error * 3 / 16
            img_array[y + 1, x] += error * 5 / 16
            img_array[y + 1, x + 1] += error * 1 / 16

    img_array[img_array >= 128] = 255
    img_array[img_array < 128] = 0

    return Image.fromarray(img_array.astype(np.uint8), mode='L').convert('1')


def pad_to_1920x1080(image):
    """
    将图像用黑色填充到1920×1080，居中显示
    """
    target_width, target_height = 1920, 1080
    padded_image = Image.new('1', (target_width, target_height), color=0)

    img_width, img_height = image.size
    x_offset = (target_width - img_width) // 2
    y_offset = (target_height - img_height) // 2

    padded_image.paste(image, (x_offset, y_offset))
    return padded_image


# 主流程
if __name__ == "__main__":
    phases = [0, 90, 180, 270]

    print("开始处理...")
    for i, phase in enumerate(phases):
        # 1. 生成128×128傅里叶基底
        basis = generate_fourier_basis(fx=3, fy=3, phase_deg=phase)

        # 2. 放大8倍（水平和垂直方向）
        enlarged_size = (basis.width * 8, basis.height * 8)
        enlarged = basis.resize(enlarged_size, Image.NEAREST)

        # 3. Floyd-Steinberg抖动二值化
        dithered = floyd_steinberg_dithering(enlarged)

        # 4. 填充到1920×1080
        final_image = pad_to_1920x1080(dithered)

        # 5. 保存为BMP格式
        filename = f'fx=fy=3_{phase}.bmp'
        # 指定保存目录
        save_dir = r'D:\桌面\毕设\picture\傅里叶基底1bit（1024×1024）'
        full_path = os.path.join(save_dir, filename)
        # 保存图片
        final_image.save(full_path)

        print(f"  相位{phase}°处理完成 -> {filename}")

    print("\n全部完成！生成了4个BMP图像")