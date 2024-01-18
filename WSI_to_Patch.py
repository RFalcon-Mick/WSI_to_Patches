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

# 创建输出目录
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建一个字典来存储坐标信息
tile_coordinates = {}

# 遍历input目录下的所有.svs文件
for file_name in os.listdir('input'):
    if file_name.endswith('.svs'):
        file_path = os.path.join('input', file_name)
        slide = openslide.OpenSlide(file_path)

        # 获取下采样倍率，这里我们假设使用第一个下采样级别
        downsample = slide.level_downsamples[1]
        level = 1  # 第一个下采样级别

        # 读取下采样后的图像
        img = slide.read_region((0, 0), level, slide.level_dimensions[level])
        img = np.array(img)[:, :, :3]  # 删除alpha通道

        # 将RGB转换为HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 在饱和度通道上应用大津二值化
        _, foreground = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 寻找前景区域的轮廓
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个黑色的图像来绘制前景区域
        foreground_mask = np.zeros_like(foreground)

        # 只保留大的前景区域
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # 可调整阈值以滤除小区域
                cv2.drawContours(foreground_mask, [contour], -1, 255, -1)

        # 将前景区域转换回原始尺寸
        foreground_mask = cv2.resize(foreground_mask, slide.level_dimensions[0], interpolation=cv2.INTER_NEAREST)

        # 分割为256x256的图像块
        for i in range(0, foreground_mask.shape[0], TILE_SIZE):
            for j in range(0, foreground_mask.shape[1], TILE_SIZE):
                tile = foreground_mask[i:i + TILE_SIZE, j:j + TILE_SIZE]
                # 检查图像块是否包含前景
                if np.sum(tile) > 0:  # 这表示图像块中有前景
                    tile_img = np.array(
                        slide.read_region((int(j * downsample), int(i * downsample)), 0, (TILE_SIZE, TILE_SIZE)))[:, :,
                               :3]
                    tile_output_path = os.path.join(output_dir,
                                                    f"{file_name}_x{j * int(downsample)}_y{i * int(downsample)}.png")
                    cv2.imwrite(tile_output_path, cv2.cvtColor(tile_img, cv2.COLOR_RGBA2RGB))
                    print(f"Saved tile to {tile_output_path}")
                    # 保存坐标信息
                    tile_coordinates[tile_output_path] = (j * int(downsample), i * int(downsample))

        slide.close()

# 将坐标信息写入一个CSV文件
coordinates_file = os.path.join(output_dir, 'patch_coordinates.csv')
with open(coordinates_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Tile Path', 'X Coordinate', 'Y Coordinate'])
    for path, (x, y) in tile_coordinates.items():
        writer.writerow([path, x, y])
