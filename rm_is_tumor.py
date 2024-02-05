# 导入os模块，用于操作文件和目录
import os

# 定义一个函数，用于创建normal和tumor文件夹，并将文件夹按照名称的14和15位进行分类
def classify_folders(source_folder):
    # 在源文件夹下创建normal和tumor文件夹
    os.mkdir(os.path.join(source_folder, "normal"))
    os.mkdir(os.path.join(source_folder, "tumor"))
    # 遍历源文件夹下的所有文件夹
    for folder in os.listdir(source_folder):
        # 如果文件夹的名称的14和15位是00~09，表示是tumor
        if folder[13:15] in ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            # 将文件夹移动到tumor文件夹中
            os.rename(os.path.join(source_folder, folder), os.path.join(source_folder, "tumor", folder))
        # 如果文件夹的名称的14和15位是10~19，表示是normal
        elif folder[13:15] in ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]:
            # 将文件夹移动到normal文件夹中
            os.rename(os.path.join(source_folder, folder), os.path.join(source_folder, "normal", folder))
        # 否则，忽略该文件夹
        else:
            pass

# 调用函数，分别对randomized文件夹下的train文件夹和vail文件夹进行操作
classify_folders("randomized/train")
classify_folders("randomized/vail")
