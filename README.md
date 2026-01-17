# WSI_to_Patches

本项目用于将 `.svs` 格式的全玻片图像（Whole Slide Image, WSI）分割成 Patch，并支持多进程/多线程并行、下采样、ROI 初筛等功能。当前版本改为使用配置文件驱动，便于记录每次实验设置与输出结果。

## 功能概览

- 批量切分 WSI 为 Patch。
- 支持下采样倍率或 MPP 目标分辨率。
- 可选过滤颜色不鲜明或黑色占比高的 Patch。
- 多进程 + 线程池并行处理。
- 支持输出格式 `jpg/png/tif` 等。
- 每次实验输出目录包含实验名称并保存配置快照。

## 目录结构

```
WSI_to_Patches/
├── config.json
├── input/          # 待处理的 WSI 文件
├── output/         # 输出根目录（实验名会生成子目录）
├── WSI_to_Patch.py # 主脚本
```

## 快速开始

1. 准备依赖（也可使用 `init_project_env.py` 协助安装 OpenSlide 环境）：
   - Python 3.x
   - Pillow, opencv-python, numpy, openslide-python, tqdm, scikit-image
2. 编辑 `config.json`，设置实验名称、输出格式等参数。
3. 运行脚本：

```
python WSI_to_Patch.py --config config.json
```

## 配置文件说明

`config.json` 使用 JSON 格式，常用字段如下：

- `experiment_name`：实验名称，输出目录会自动带上该名称。
- `tile_size`：Patch 尺寸，支持 128/256/512/1024。
- `num_threads`：线程数，支持 1/2/4/8。
- `processes`：进程数，支持 1/2/4/8。
- `downsample_factor`：下采样倍率（1/2/4/8/16/32），与 `mpp` 二选一。
- `mpp`：目标 MPP，自动推算下采样倍率（可选，若为非整数倍率将自动重采样）。
- `input_dir`：WSI 输入目录。
- `output_dir`：输出根目录（实际输出会附加实验名称）。
- `filter`：是否过滤颜色不鲜明或黑色占比高的 Patch。
- `output_format`：输出格式（`jpg`/`png`/`tif`/`tiff`）。

当 `downsample_factor` 与 `mpp` 均未设置时，将默认使用 2 倍下采样。

## 输出结构

```
output/
└── <experiment_name>/
    ├── config.json
    ├── <wsi_name_1>/
    │   ├── <wsi_name_1>_x0_y0.png
    │   └── ...
    └── <wsi_name_2>/
        └── ...
```

每次实验都会在输出目录中保存一份 `config.json` 作为参数记录。

## 注意事项

1. 使用 `mpp` 时要求 WSI 中包含 MPP 元数据，否则请使用 `downsample_factor`。非整数倍率会自动采用重采样。
2. Windows 用户需确保 OpenSlide DLL 可用，Linux 用户可使用系统包管理器安装 OpenSlide。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 联系方式

如有问题或建议，请联系：
- 邮箱：dev@rfalcon.cn
- GitHub：[https://github.com/RFalcon-Mick](https://github.com/RFalcon-Mick)
