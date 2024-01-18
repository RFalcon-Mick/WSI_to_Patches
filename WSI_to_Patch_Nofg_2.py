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
TILE_SIZE = 256
# 定义线程数量
NUM_THREADS = 8

# 创建输出目录
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建一个字典来存储坐标信息
tile_coordinates = {}

def save_tile(tile_img, x, y, file_name):
    """保存图像块和其坐标"""
    tile_output_path = os.path.join(output_dir, f"{file_name}_x{x}_y{y}.png")
    cv2.imwrite(tile_output_path, cv2.cvtColor(tile_img, cv2.COLOR_RGBA2RGB))
    print(f"Saved tile to {tile_output_path}")
    return tile_output_path, (x, y)

# 处理单个文件
def process_file(file_name):
    file_path = os.path.join('input', file_name)
    slide = openslide.OpenSlide(file_path)
    downsample = slide.level_downsamples[1]
    level = 1  # 第一个下采样级别
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)[:, :, :3]  # 删除alpha通道

    # 使用线程池来保存图像块
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for i in range(0, img.shape[0], TILE_SIZE):
            for j in range(0, img.shape[1], TILE_SIZE):
                tile = img[i:i + TILE_SIZE, j:j + TILE_SIZE]
                if np.any(tile != 0):  # 检查图像块是否不全是黑色
                    tile_img = np.array(slide.read_region((int(j * downsample), int(i * downsample)), 0, (TILE_SIZE, TILE_SIZE)))[:, :, :3]
                    future = executor.submit(save_tile, tile_img, j * int(downsample), i * int(downsample), file_name)
                    futures.append(future)
        for future in futures:
            path, coords = future.result()
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
