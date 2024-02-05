# 导入必要的模块
import os
import shutil
import random # 导入random模块

# 询问用户测试集和训练集的比例
train_ratio = float(input("请输入训练集的比例（0~1）："))
test_ratio = 1-train_ratio


# 询问用户随机种子
seed = input("请输入一个整数作为随机种子，如果不想设置，请输入None\n")
random.seed(seed) # 设置随机种子

# 定义源目录和目标目录
source_dir = "output"
target_dir = "randomized"
tumor_dir = os.path.join(source_dir, "tumor")
normal_dir = os.path.join(source_dir, "normal")
train_dir = os.path.join(target_dir, "train")
test_dir = os.path.join(target_dir, "vail")

# 如果目标目录不存在，就创建它
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# 定义一个函数，实现按照比例随机复制子目录和文件
def copy_subdirs(source, target, test_ratio, train_ratio):
    # 获取源目录下的所有子目录
    subdirs = os.listdir(source)
    # 遍历每个子目录
    for subdir in subdirs:
        # 获取子目录的完整路径
        subdir_path = os.path.join(source, subdir)
        # 判断是否是目录，如果是，就继续处理
        if os.path.isdir(subdir_path):
            # 生成一个随机数，根据比例判断是复制到测试集还是训练集
            rand = random.random()
            if rand < test_ratio:
                # 复制到测试集
                target_subdir = os.path.join(test_dir, subdir)
            else:
                # 复制到训练集
                target_subdir = os.path.join(train_dir, subdir)
            # 如果目标子目录不存在，就创建它
            if not os.path.exists(target_subdir):
                os.mkdir(target_subdir)
            # 获取子目录下的所有文件
            files = os.listdir(subdir_path)
            # 遍历每个文件，复制到目标子目录
            for file in files:
                file_path = os.path.join(subdir_path, file)
                target_file = os.path.join(target_subdir, file)
                shutil.copy(file_path, target_file)
                # 打印文件名
                print(f"已复制文件{file}")

# 调用函数，分别处理tumor和normal目录
copy_subdirs(tumor_dir, target_dir, test_ratio, train_ratio)
copy_subdirs(normal_dir, target_dir, test_ratio, train_ratio)

# 打印完成提示
print("随机复制完成！")
