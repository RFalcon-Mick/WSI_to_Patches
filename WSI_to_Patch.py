import argparse
import cv2
import json
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
OUTPUT_FORMAT_CHOICES = {"jpg", "jpeg", "png", "tif", "tiff"}
OUTPUT_FORMAT_ALIASES = {"jpeg": "jpg"}
DEFAULT_CONFIG = {
    "experiment_name": "default_experiment",
    "tile_size": 512,
    "num_threads": 2,
    "downsample_factor": None,
    "mpp": None,
    "input_dir": "input",
    "output_dir": "output",
    "processes": 4,
    "filter": False,
    "output_format": "png",
}

# 定义命令行参数
parser = argparse.ArgumentParser(description='Extract and save tiles from .svs image files.')
parser.add_argument(
    '-c',
    '--config',
    type=Path,
    default=Path('config.json'),
    help='Path to the JSON configuration file.'
)


def load_config(config_path: Path):
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            config_data = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file {config_path} is not valid JSON: {exc}") from exc

    config = DEFAULT_CONFIG.copy()
    config.update(config_data)

    experiment_name = str(config.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError("experiment_name must be a non-empty string.")
    config["experiment_name"] = experiment_name

    tile_size = config.get("tile_size")
    if tile_size is None:
        tile_size = DEFAULT_CONFIG["tile_size"]
    tile_size = int(tile_size)
    if tile_size not in [128, 256, 512, 1024]:
        raise ValueError("tile_size must be one of [128, 256, 512, 1024].")
    config["tile_size"] = tile_size

    num_threads = config.get("num_threads")
    if num_threads is None:
        num_threads = DEFAULT_CONFIG["num_threads"]
    num_threads = int(num_threads)
    if num_threads not in [1, 2, 4, 8]:
        raise ValueError("num_threads must be one of [1, 2, 4, 8].")
    config["num_threads"] = num_threads

    processes = config.get("processes")
    if processes is None:
        processes = DEFAULT_CONFIG["processes"]
    processes = int(processes)
    if processes not in [1, 2, 4, 8]:
        raise ValueError("processes must be one of [1, 2, 4, 8].")
    config["processes"] = processes

    downsample_factor = config.get("downsample_factor")
    if downsample_factor is not None:
        downsample_factor = int(downsample_factor)
        if downsample_factor not in DOWNSAMPLE_CHOICES:
            raise ValueError(f"downsample_factor must be one of {DOWNSAMPLE_CHOICES}.")
    config["downsample_factor"] = downsample_factor

    mpp = config.get("mpp")
    if mpp is not None:
        mpp = float(mpp)
        if mpp <= 0:
            raise ValueError("mpp must be a positive number.")
    config["mpp"] = mpp

    if config["downsample_factor"] is None and config["mpp"] is None:
        config["downsample_factor"] = DEFAULT_DOWNSAMPLE_FACTOR

    output_format = config.get("output_format")
    if output_format is None:
        output_format = DEFAULT_CONFIG["output_format"]
    output_format = str(output_format).lower()
    if output_format.startswith("."):
        output_format = output_format[1:]
    if output_format not in OUTPUT_FORMAT_CHOICES:
        raise ValueError(f"output_format must be one of {sorted(OUTPUT_FORMAT_CHOICES)}.")
    config["output_format"] = OUTPUT_FORMAT_ALIASES.get(output_format, output_format)

    config["filter"] = bool(config.get("filter", False))
    input_dir = Path(config.get("input_dir") or DEFAULT_CONFIG["input_dir"])
    output_base_dir = Path(config.get("output_dir") or DEFAULT_CONFIG["output_dir"])
    experiment_output_dir = output_base_dir / experiment_name

    config["input_dir"] = input_dir
    config["output_dir"] = experiment_output_dir

    config_snapshot = {
        **config,
        "input_dir": str(input_dir),
        "output_dir": str(experiment_output_dir),
        "config_path": str(config_path),
    }
    return config, config_snapshot


args = parser.parse_args()
CONFIG_SNAPSHOT = {}
try:
    args_config, CONFIG_SNAPSHOT = load_config(args.config)
except ValueError as exc:
    parser.error(str(exc))

args = argparse.Namespace(**args_config)

# 创建输出目录
args.output_dir.mkdir(parents=True, exist_ok=True)

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
    if mpp_x is None or mpp_y is None:
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
    slide_name = file_path.stem
    slide = openslide.OpenSlide(str(file_path))
    downsample_factor = resolve_downsample_factor(slide, file_name)

    # 创建一个子目录，名称为slide_name
    sub_dir = args.output_dir.joinpath(slide_name)
    sub_dir.mkdir(parents=True, exist_ok=True)

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
            result = pool.apply_async(save_tile, args=(tile_img_downsampled, i, j, slide_name, sub_dir))
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


def save_tile(tile_img, x, y, slide_name, sub_dir):
    """保存图像块和其坐标"""
    # 修改tile_output_path，让它指向子目录
    tile_output_path = sub_dir.joinpath(f"{slide_name}_x{x}_y{y}.{args.output_format}")
    if tile_img.shape[-1] == 4:
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2BGR)
    else:
        tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tile_output_path), tile_img)
    return tile_output_path, (x, y)


def save_config_snapshot():
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir.joinpath("config.json")
    try:
        with config_path.open('w', encoding='utf-8') as config_file:
            json.dump(CONFIG_SNAPSHOT, config_file, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise RuntimeError(f"Failed to save config snapshot to {config_path}: {exc}") from exc


if __name__ == '__main__':
    mp.freeze_support()
    save_config_snapshot()
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
