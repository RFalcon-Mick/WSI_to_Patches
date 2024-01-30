# WSI_to_Patches
本项目旨在将svs格式的全玻片图像（Whole Slide Image, WSI）分割成多个小块（Patch），同时自动去除前景图像块。此外，项目还包含一个可选脚本，可以根据RGB三个通道的值来判断图像中病理学切片占比，从而进行筛选。最新更新中，我们对项目进行了以下优化和扩展：
- 优化了判断图像是否颜色鲜明的算法。
- 重构了分割、下采样和保存相关代码，提高了处理效率。
- 重构了代码执行逻辑，使流程更加清晰。
- 实现了多进程、线程并行处理，大幅提升了处理速度。
- 新增了命令行控制参数功能，方便用户根据需求自定义处理流程。
- 完善了图片初步过滤筛选功能，提升了ROI筛选的准确性。
## 使用方法
1. 克隆本项目到本地。
2. 将svs格式的WSI图像文件放入`input`文件夹。
3. 运行`WSI_to_Patch.py`。该脚本会自动将WSI图像分割成Patch，并生成相对应的坐标，存储在`output`文件夹中。现在可以通过命令行参数自定义处理流程。
## 文件结构
- `input/`: 存放待处理的svs格式WSI图像。
- `output/`: 存放分割后的Patch图像及其坐标。
- `WSI_to_Patch.py`: 将WSI图像分割成Patch。
## 注意事项
1. 在运行`WSI_to_Patch.py`之前，请确保已经安装了Python和pip。
2. 使用命令行参数时，请参考`WSI_to_Patch.py`中的帮助信息进行操作。
## 许可证
本项目采用[MIT许可证](LICENSE)。
## 联系方式
如果您在使用本项目过程中遇到任何问题，或有任何建议，欢迎通过以下方式与我们联系：
- 邮箱：dev@rfalcon.cn
- GitHub：[https://github.com/RFalcon-Mick](https://github.com/RFalcon-Mick)

感谢您对项目的关注，我们期待您的反馈和建议。
