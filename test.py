from concurrent.futures import ThreadPoolExecutor
import tqdm
#TODO tqdm是新加入的包，需要适配init
import argparse
import cv2
import numpy as np
import csv
import logging
from pathlib import Path
import os
with os.add_dll_directory(str(Path(__file__).parent.joinpath('openslide', 'bin'))):
    import openslide
# 导入multiprocessing模块
import multiprocessing as mp
# 导入tqdm模块
from tqdm import tqdm
# 导入PIL模块
from PIL import Image
# 导入shutil模块
import shutil

# 设置日志格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义命令行参数
parser = argparse.ArgumentParser(description='Extract and save tiles from .svs image files.')
parser.add_argument('-t', '--tile_size', type=int, choices=[128, 256, 512, 1024], default=256, help='The size of the tile in pixels.')
parser.add_argument('-n', '--num_threads', type=int, choices=[1, 2, 4, 8], default=1, help='The number of threads to use.')
parser.add_argument('-d', '--downsample_factor', type=int, choices=[1, 2, 4, 8, 16, 32], default=2, help='The factor to downsample the image by.')
parser.add_argument('-i', '--input_dir', type=Path, default=Path('input'), help='The directory where the .svs files are located.')
parser.add_argument('-o', '--output_dir', type=Path, default=Path('output'), help='The directory where the tiles and coordinates will be saved.')
parser.add_argument('-p', '--processes', type=int, choices=[1, 2, 4, 8], default=8, help='The number of processes to use.')
# 添加一个新的参数，用于控制是否调用第二段代码
parser.add_argument('-f', '--filter', action='store_true', help='Whether to filter out the images that are not colorful or have large black areas.')
args = parser.parse_args()

# 创建输出目录
args.output_dir.mkdir(exist_ok=True)

# 创建一个字典来存储坐标信息
tile_coordinates = {}

def is_tile_mostly_black(tile_img, threshold=0.4):
    """判断图像块是否有超过一定比例的黑色像素"""
    gray_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2GRAY)
    black_pixels = np.sum(gray_img == 0)
    total_pixels = gray_img.size
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold

def process_file(file_path):
    file_name = file_path.name
    slide = openslide.OpenSlide(str(file_path))
    downsample = slide.level_downsamples[1] * args.downsample_factor
    level = slide.get_best_level_for_downsample(downsample)
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)[:, :, :3]  # 删除alpha通道

    # 创建一个子目录，名称为file_name
    sub_dir = args.output_dir.joinpath(file_name)
    sub_dir.mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        tile_count = 0 # 记录处理的图像块的数量
        for i in range(0, img.shape[0], args.tile_size):
            for j in range(0, img.shape[1], args.tile_size):
                tile = img[i:i + args.tile_size, j:j + args.tile_size]
                if np.any(tile != 0):  # 检查图像块是否不全是黑色
                    tile_img = np.array(slide.read_region((int(j * downsample), int(i * downsample)), 0, (args.tile_size, args.tile_size)))[:, :, :3]
                    if not is_tile_mostly_black(tile_img):
                        future = executor.submit(save_tile, tile_img, j * int(downsample), i * int(downsample), file_name, sub_dir)
                        futures.append(future)
                        tile_count += 1 # 增加图像块的数量
        # 添加一个tqdm对象，来迭代futures列表，并设置一个描述和一个总数
        for future in tqdm(futures, desc=f"Processing {file_name}", total=tile_count):
            result = future.result()
            if result:
                path, coords = result
                tile_coordinates[path] = coords

    slide.close()
    # 返回文件名和图像块的数量的元组
    return file_name, tile_count

def save_tile(tile_img, x, y, file_name, sub_dir):
    """保存图像块和其坐标"""
    if is_tile_mostly_black(tile_img):
        # 删除logger.info语句，因为它们会干扰进度条的显示
        # logger.info(f"Tile at x={x}, y={y} is mostly black and will be discarded.")
        return None
    else:
        # 修改tile_output_path，让它指向子目录
        tile_output_path = sub_dir.joinpath(f"{file_name}_x{x}_y{y}.png")
        cv2.imwrite(str(tile_output_path), cv2.cvtColor(tile_img, cv2.COLOR_RGBA2RGB))
        return tile_output_path, (x, y)

def is_colorful(image_path, threshold=200, black_threshold=0.05):
    """判断图像是否颜色鲜明且不包含大块黑色"""
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
    """复制颜色鲜明且不包含大块黑色的图像到输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            if is_colorful(image_path):
                shutil.copy2(image_path, output_dir)
                print(f"成功复制图像: {filename}")

# 在主模块中使用if __name__ == '__main__'语句
# 在主模块中使用if __name__ == '__main__'语句
if __name__ == '__main__':
    # 使用freeze_support()函数
    mp.freeze_support()
    # 创建一个进程池，进程数可以根据你的CPU核心数调整
    pool = mp.Pool(processes=args.processes)

    # 遍历input目录下的所有.svs文件
    results = [] # 创建一个列表，用于存储AsyncResult对象
    total_files = 0 # 记录总的文件数
    for file_path in args.input_dir.iterdir():
        if file_path.is_file() and file_path.suffix == '.svs':
            # 将process_file函数作为任务提交给进程池
            result = pool.apply_async(process_file, args=(file_path,))
            results.append(result) # 将AsyncResult对象添加到列表中
            total_files += 1 # 增加文件数

    # 创建一个tqdm对象，用于显示总的处理进度
    pbar = tqdm(total=total_files, desc="Total progress")

    # 在主进程中不断地检查每个任务的状态，并更新进度条
    while True:
        # 如果所有的任务都完成了，就跳出循环
        if all(result.ready() for result in results):
            break
        # 否则，遍历每个任务，如果有完成的，就更新进度条，并打印文件名和图像块的数量
        for result in results:
            if result.ready():
                file_name, tile_count = result.get()
                pbar.update(1)
                print(f"Finished {file_name} with {tile_count} tiles")
                results.remove(result)  # 从列表中移除已完成的任务

    # 关闭进程池，等待所有进程完成
    pool.close()
    pool.join()
    # 将坐标信息写入一个CSV文件，用于保存每个图像块在原始图像中的位置
    coordinates_file = args.output_dir.joinpath('patch_coordinates.csv')
    with open(coordinates_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tile Path', 'X Coordinate', 'Y Coordinate'])
        for path, (x, y) in tile_coordinates.items():
            writer.writerow([path, x, y])

    # 如果使用了-f --filter开关，就调用第二段代码
    if args.filter:
        # 设置输入和输出目录
        input_dir = args.output_dir
        output_dir = args.output_dir.joinpath('processed_output')
        # 复制颜色鲜明且不包含大块黑色的图像到输出目录
        copy_colorful_images(input_dir, output_dir)
