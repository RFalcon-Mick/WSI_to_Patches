from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import numpy as np
import csv
current_working_directory = os.getcwd()
relative_path = r"\openslide\bin"
absolute_path = current_working_directory + relative_path
OPENSLIDE_PATH = absolute_path
with os.add_dll_directory(OPENSLIDE_PATH):
    import openslide

# 定义要处理的图像块大小
print("请选择Patch的大小：（单位:px）")
print("1. 128")
print("2. 256")
print("3. 512")
print("4. 1024")
choice = input("请输入你的选择（1-4）: ")
tile_size_mapping = {'1': 128, '2': 256, '3': 512, '4': 1024}
TILE_SIZE = tile_size_mapping.get(choice, 256)  # 默认值256
print(f"Patch的大小设定为: {TILE_SIZE}")

# 定义线程数量
print("请选择线程数：")
print("1. 1")
print("2. 2")
print("3. 4")
print("4. 8")
choice = input("请输入你的选择（1-4）: ")
num_threads_mapping = {'1': 1, '2': 2, '3': 4, '4': 8}
NUM_THREADS = num_threads_mapping.get(choice, 1)  # 默认值1
print(f"线程数设定为: {NUM_THREADS}")

# 让用户选择下采样倍率
print("请选择下采样倍率：")
print("1. 不下采样")
print("2. 2倍")
print("3. 4倍")
print("4. 8倍")
print("5. 16倍")
print("6. 32倍")
downsample_choice = input("请输入你的选择（1-6）: ")
downsample_mapping = {'1': 1, '2': 2, '3': 4, '4': 8, '5': 16, '6': 32}
DOWNSAMPLE_FACTOR = downsample_mapping.get(downsample_choice, 1)  # 默认值2倍
print(f"下采样倍率设定为: {DOWNSAMPLE_FACTOR}倍")

# 创建输出目录
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建一个字典来存储坐标信息
tile_coordinates = {}

def is_tile_mostly_black(tile_img, threshold=0.4):
    """判断图像块是否有超过一定比例的黑色像素"""
    gray_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2GRAY)
    black_pixels = np.sum(gray_img == 0)
    total_pixels = gray_img.size
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold

def save_tile(tile_img, x, y, file_name):
    """保存图像块和其坐标"""
    if is_tile_mostly_black(tile_img):
        print(f"Tile at x={x}, y={y} is mostly black and will be discarded.")
        return None
    else:
        tile_output_path = os.path.join(output_dir, f"{file_name}_x{x}_y{y}.png")
        cv2.imwrite(tile_output_path, cv2.cvtColor(tile_img, cv2.COLOR_RGBA2RGB))
        print(f"Saved tile to {tile_output_path}")
        return tile_output_path, (x, y)

def process_file(file_name):
    file_path = os.path.join('input', file_name)
    slide = openslide.OpenSlide(file_path)
    downsample = slide.level_downsamples[1] * DOWNSAMPLE_FACTOR
    level = slide.get_best_level_for_downsample(downsample)
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)[:, :, :3]  # 删除alpha通道

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i in range(0, img.shape[0], TILE_SIZE):
            for j in range(0, img.shape[1], TILE_SIZE):
                tile = img[i:i + TILE_SIZE, j:j + TILE_SIZE]
                if np.any(tile != 0):  # 检查图像块是否不全是黑色
                    tile_img = np.array(slide.read_region((int(j * downsample), int(i * downsample)), 0, (TILE_SIZE, TILE_SIZE)))[:, :, :3]
                    if not is_tile_mostly_black(tile_img):
                        future = executor.submit(save_tile, tile_img, j * int(downsample), i * int(downsample), file_name)
                        futures.append(future)
        for future in futures:
            result = future.result()
            if result:
                path, coords = result
                tile_coordinates[path] = coords

    slide.close()

# 遍历input目录下的所有.svs文件
for file_name in os.listdir('input'):
    if file_name.endswith('.svs'):
        process_file(file_name)

# 将坐标信息写入一个CSV文件
coordinates_file = os.path.join(output_dir, 'patch_coordinates.csv')
with open(coordinates_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tile Path', 'X Coordinate', 'Y Coordinate'])
    for path, (x, y) in tile_coordinates.items():
        writer.writerow([path, x, y])
