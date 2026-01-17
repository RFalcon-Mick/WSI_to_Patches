import argparse
import cv2
import math
import multiprocessing as mp
import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import measure

with os.add_dll_directory(str(Path(__file__).parent.joinpath('openslide', 'bin'))):
    import openslide

DOWNSAMPLE_CHOICES = [1, 2, 4, 8, 16, 32]
DEFAULT_DOWNSAMPLE_FACTOR = 2
MPP_REL_TOLERANCE = 1e-3
MPP_ABS_TOLERANCE = 1e-6

# 定义命令行参数
parser = argparse.ArgumentParser(description='Extract and save tiles from .svs image files.')
parser.add_argument('-t', '--tile_size', type=int, choices=[128, 256, 512, 1024], default=512,
                    help='The size of the tile in pixels.')
parser.add_argument('-n', '--num_threads', type=int, choices=[1, 2, 4, 8], default=2,
                    help='The number of threads to use.')
parser.add_argument('-d', '--downsample_factor', '--sampling_rate', '--sample_rate', type=int,
                    choices=DOWNSAMPLE_CHOICES, default=None,
                    help='The factor to downsample the image by (sampling rate).')
parser.add_argument('--mpp', type=float, default=None,
                    help='Target microns-per-pixel (MPP) to derive downsampling from slide metadata.')
parser.add_argument('-i', '--input_dir', '--wsi_dir', type=Path, default=Path('input'),
                    help='The directory where the WSI .svs files are located.')
parser.add_argument('-o', '--output_dir', '--out_dir', type=Path, default=Path('output'),
                    help='The directory where the tiles and coordinates will be saved.')
parser.add_argument('-p', '--processes', type=int, choices=[1, 2, 4, 8], default=4,
                    help='The number of processes to use.')
parser.add_argument('-f', '--filter', action='store_true',
                    help='Whether to filter out the images that are not colorful or have large black areas.')
args = parser.parse_args()

if args.mpp is not None and args.mpp <= 0:
    parser.error('--mpp must be a positive number.')

if args.downsample_factor is None and args.mpp is None:
    args.downsample_factor = DEFAULT_DOWNSAMPLE_FACTOR

# 创建输出目录
args.output_dir.mkdir(exist_ok=True)

# 创建一个字典来存储坐标信息
tile_coordinates = {}


# 定义一个函数，判断图像是否颜色鲜明且不包含大块黑色
def is_colorful(image, threshold=200, black_threshold=0.05):
    """判断图像是否颜色鲜明且不包含大块黑色"""
    # 如果是路径，就打开图像
    if isinstance(image, Path):
        image = Image.open(image)
    # 如果是numpy数组，就转换成图像
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # 转换为RGB模式
    rgb = image.convert("RGB")
    rgb = np.array(rgb)  # 添加这一行
    width, height = image.size

    # 使用numpy的sum函数来计算总的RGB值和黑色像素的数量，这样比逐个像素遍历要快
    total_r = np.sum(rgb[:, :, 0])
    total_g = np.sum(rgb[:, :, 1])
    total_b = np.sum(rgb[:, :, 2])
    black_pixels = np.sum(rgb[:, :, 0] == 0) + np.sum(rgb[:, :, 1] == 0) + np.sum(rgb[:, :, 2] == 0)

    avg_r = total_r // (width * height)
    avg_g = total_g // (width * height)
    avg_b = total_b // (width * height)

    return avg_r < threshold and avg_g < threshold and avg_b < threshold and black_pixels / (
            width * height) < black_threshold


def resolve_downsample_factor(slide, file_name):
    if args.mpp is None:
        return args.downsample_factor

    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if not mpp_x or not mpp_y:
        raise ValueError(
            f"{file_name} is missing MPP metadata; rerun with --downsample_factor instead of --mpp."
        )

    base_mpp = (float(mpp_x) + float(mpp_y)) / 2
    if base_mpp <= 0:
        raise ValueError(
            f"{file_name} has invalid MPP metadata; rerun with --downsample_factor instead of --mpp."
        )

    computed = args.mpp / base_mpp
    rounded = int(round(computed))
    if not math.isclose(
        computed,
        rounded,
        rel_tol=MPP_REL_TOLERANCE,
        abs_tol=MPP_ABS_TOLERANCE
    ):
        raise ValueError(
            f"{file_name} needs a non-integer downsample factor ({computed:.3f}) for MPP {args.mpp}; "
            "use --downsample_factor instead."
        )

    if rounded not in DOWNSAMPLE_CHOICES:
        raise ValueError(
            f"{file_name} computed downsample factor {rounded} from MPP {args.mpp}, which is unsupported. "
            f"Choose one of {DOWNSAMPLE_CHOICES} or use --downsample_factor."
        )

    if args.downsample_factor is not None and args.downsample_factor != rounded:
        raise ValueError(
            f"{file_name} downsample_factor {args.downsample_factor} conflicts with MPP-derived {rounded} "
            f"(target MPP {args.mpp} vs base MPP {base_mpp:.3f}). Remove one option or make them consistent."
        )

    return rounded


def process_file(file_path):
    file_name = file_path.name
    slide = openslide.OpenSlide(str(file_path))
    downsample_factor = resolve_downsample_factor(slide, file_name)

    # 创建一个子目录，名称为file_name
    sub_dir = args.output_dir.joinpath(file_name)
    sub_dir.mkdir(exist_ok=True)

    # 使用ThreadPool来创建线程池
    pool = ThreadPool(args.num_threads)
    results = []
    tile_count = 0  # 记录处理的图像块的数量

    for i in np.arange(0, slide.dimensions[0], args.tile_size):
        for j in np.arange(0, slide.dimensions[1], args.tile_size):
            tile_img = np.array(slide.read_region((i, j), 0, (args.tile_size, args.tile_size)))[:, :, :3]

            # 进行下采样和保存
            tile_img_downsampled = measure.block_reduce(tile_img, (downsample_factor, downsample_factor, 1), np.mean)
            tile_img_downsampled = tile_img_downsampled.astype(np.uint8)
            result = pool.apply_async(save_tile, args=(tile_img_downsampled, i, j, file_name, sub_dir))
            results.append(result)
            tile_count += 1  # 增加图像块的数量

    # 关闭线程池，等待所有线程完成
    pool.close()
    pool.join()

    # 添加一个tqdm对象，来迭代results列表，并设置一个描述和一个总数
    process_temp = tqdm(results, desc=f"Processing {file_name}", total=tile_count)

    for result in process_temp:
        result = result.get()
        process_temp.update()

        if result:
            path, coords = result
            tile_coordinates[path] = coords

            # 如果使用了-f --filter开关，就对图像进行过滤
            if args.filter and not is_colorful(path, black_threshold=0.4):
                os.remove(path)

    slide.close()
    # 返回文件名和图像块的数量的元组
    return file_name, tile_count


def save_tile(tile_img, x, y, file_name, sub_dir):
    """保存图像块和其坐标"""
    # 修改tile_output_path，让它指向子目录
    tile_output_path = sub_dir.joinpath(f"{file_name}_x{x}_y{y}.png")
    cv2.imwrite(str(tile_output_path), cv2.cvtColor(tile_img, cv2.COLOR_RGBA2RGB))
    return tile_output_path, (x, y)


if __name__ == '__main__':
    mp.freeze_support()
    # 创建一个进程池，进程数可以调整
    pool = mp.Pool(processes=args.processes)

    # 使用列表推导式来生成results列表
    results = [pool.apply_async(process_file, args=(file_path,)) for file_path in args.input_dir.iterdir() if file_path.is_file() and file_path.suffix == '.svs']
    total_files = len(results)

    # 创建一个tqdm对象，用于显示总的处理进度
    pbar = tqdm(total=total_files, desc="Total progress", position=0, leave=True, miniters=1)

    # 在主进程中不断地检查每个任务的状态，并更新进度条
    while True:
        if all(result.ready() for result in results):
            break
        for result in results:
            if result.ready():
                file_name, tile_count = result.get()
                pbar.update(1)
                results.remove(result)

    pool.close()
    pool.join()
