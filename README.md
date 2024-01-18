# WSI_to_Patches
本项目旨在将svs格式的全玻片图像（Whole Slide Image, WSI）分割成多个小块（Patch），同时自动去除前景图像块。此外，项目还包含一个可选脚本，可以根据RGB三个通道的值来判断图像中病理学切片占比，从而进行筛选。
## 使用方法
1. 克隆本项目到本地。
2. 运行`init_project_env.py`来初始化项目环境。在运行前，请确保已经安装了Python和pip。如果需要，可以在`init_project_env.py`中修改openslide环境的下载地址。
3. 将svs格式的WSI图像文件放入`input`文件夹。
4. 运行`WSI_to_Patch_Nofg_2.py`。该脚本会自动将WSI图像分割成Patch，并生成相对应的坐标，存储在`output`文件夹中。
5. 运行`png_process.py`。该脚本会从`output`文件夹中筛选出ROI图像，并保存在`processed_output`文件夹中。如果需要，可以根据实际情况修改筛选ROI图像的算法。
## 文件结构
- `input/`: 存放待处理的svs格式WSI图像。
- `output/`: 存放分割后的Patch图像及其坐标。
- `processed_output/`: 存放筛选后的ROI图像。
- `init_project_env.py`: 初始化项目环境。
- `WSI_to_Patch_Nofg_2.py`: 将WSI图像分割成Patch。
- `png_process.py`: 筛选ROI图像。
## 注意事项
1. 在运行`init_project_env.py`之前，请确保已经安装了Python和pip。
2. 如果需要修改openslide环境的下载地址，请在`init_project_env.py`中进行修改。
3. 如果需要修改筛选ROI图像的算法，请在`png_process.py`中进行修改。
## 许可证
本项目采用[MIT许可证](LICENSE)。
## 联系方式
如果您在使用本项目过程中遇到任何问题，或有任何建议，欢迎通过以下方式与我联系：
- 邮箱：mick@rfalcon.cn
- GitHub：[https://github.com/RFalcon-Mick](https://github.com/RFalcon-Mick)
