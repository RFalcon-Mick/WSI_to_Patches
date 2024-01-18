from PIL import Image
import os
import shutil

def is_colorful(image_path, threshold=200, black_threshold=0.05):
    image = Image.open(image_path)
    rgb = image.convert("RGB")
    width, height = image.size
    total_r, total_g, total_b = 0, 0, 0
    black_pixels = 0

    for x in range(width):
        for y in range(height):
            r, g, b = rgb.getpixel((x, y))
            total_r += r
            total_g += g
            total_b += b
            if r == 0 and g == 0 and b == 0:
                black_pixels += 1

    avg_r = total_r // (width * height)
    avg_g = total_g // (width * height)
    avg_b = total_b // (width * height)

    return avg_r < threshold and avg_g < threshold and avg_b < threshold and black_pixels / (width * height) < black_threshold

def copy_colorful_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            if is_colorful(image_path):
                shutil.copy2(image_path, output_dir)
                print(f"成功复制图像: {filename}")

# 设置输入和输出目录
input_dir = "output"
output_dir = "processed_output"

# 复制颜色鲜明且不包含大块黑色的图像到输出目录
copy_colorful_images(input_dir, output_dir)
