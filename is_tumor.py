# 导入os模块，用于操作文件和文件夹
import os

# 定义output文件夹的路径
output_path = "output"

# 定义tumor和normal文件夹的路径
tumor_path = os.path.join(output_path, "tumor")
normal_path = os.path.join(output_path, "normal")

# 如果tumor和normal文件夹不存在，就创建它们
if not os.path.exists(tumor_path):
    os.mkdir(tumor_path)
if not os.path.exists(normal_path):
    os.mkdir(normal_path)

# 遍历output文件夹下的所有文件夹
for folder in os.listdir(output_path):
    # 获取文件夹的完整路径
    folder_path = os.path.join(output_path, folder)
    # 如果是文件夹，就继续处理
    if os.path.isdir(folder_path):
        # 获取文件夹名称的第14和15位字符
        code = folder[13:15]
        # 如果是01~09，就将文件夹移动到tumor文件夹下
        if code in ["01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            os.rename(folder_path, os.path.join(tumor_path, folder))
        # 如果是10~19，就将文件夹移动到normal文件夹下
        elif code in ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]:
            os.rename(folder_path, os.path.join(normal_path, folder))
        # 否则，就忽略该文件夹
        else:
            pass