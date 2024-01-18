import os
import subprocess
import requests
import zipfile
from zipfile import ZipFile
from io import BytesIO
import shutil
def create_folder_if_not_exists(folder_name):
    # 检查文件夹是否存在
    if not os.path.exists(folder_name):
        # 创建文件夹
        os.makedirs(folder_name)
        print(f"已创建文件夹: {folder_name}")
    else:
        print(f"文件夹已存在: {folder_name}")
def move_contents_to_openslide(openslide_dir):
    # 检查openslide目录是否存在
    if not os.path.isdir(openslide_dir):
        print(f"目录 '{openslide_dir}' 不存在。")
        return
    # 获取openslide目录下的所有下一级目录
    subdirs = [dir for dir in os.listdir(openslide_dir) if os.path.isdir(os.path.join(openslide_dir, dir))]
    # 遍历所有下一级目录
    for subdir in subdirs:
        subdir_path = os.path.join(openslide_dir, subdir)
        # 获取下一级目录中的所有文件和文件夹
        contents = os.listdir(subdir_path)
        # 遍历所有内容
        for item in contents:
            item_path = os.path.join(subdir_path, item)
            # 目标路径
            target_path = os.path.join(openslide_dir, item)
            # 如果是文件，直接移动
            if os.path.isfile(item_path):
                shutil.move(item_path, target_path)
            # 如果是文件夹，递归地移动内容
            elif os.path.isdir(item_path):
                shutil.move(item_path, target_path)


def unzip_to_openslide(zip_file_path, extract_dir):
    # 确保提取目录存在
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    # 打开压缩文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 解压所有文件到指定目录
        zip_ref.extractall(extract_dir)
def find_and_unzip(openslide_dir):
    # 检查openslide目录是否存在
    if not os.path.isdir(openslide_dir):
        print(f"目录 '{openslide_dir}' 不存在。")
        return
    # 遍历目录下的所有文件
    for file in os.listdir(openslide_dir):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(openslide_dir, file)
            print(f"找到压缩包: {zip_file_path}")
            unzip_to_openslide(zip_file_path, openslide_dir)
            print(f"已解压 '{file}' 到 '{openslide_dir}' 目录。")


def download_file(url, destination_folder):
    # 发送HTTP请求
    response = requests.get(url)
    # 检查请求是否成功
    if response.status_code == 200:
        # 从响应中获取内容
        zip_content = BytesIO(response.content)
        # 创建ZipFile对象
        with ZipFile(zip_content) as zip_file:
            # 解压到指定目录
            zip_file.extractall(destination_folder)
        print(f"文件已下载并解压到 '{destination_folder}' 目录。")
    else:
        print("下载失败，请检查网络连接或URL是否正确。")

def main():
    # 定义文件夹名称列表
    folders = ['input', 'output', 'processed_output']
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 遍历文件夹列表，创建不存在的文件夹
    for folder in folders:
        folder_path = os.path.join(current_dir, folder)
        create_folder_if_not_exists(folder_path)

    try:
        subprocess.check_call(["pip", "install", "--upgrade", "pip"])
        print("pip 已更新")
    except subprocess.CalledProcessError as e:
        print("pip 更新失败")
        print(f"错误信息: {e}")

    # 要安装的包列表
    packages_to_install = ['Pillow', 'opencv-python', 'numpy', 'openslide-python']

    # 尝试安装每个包
    for package in packages_to_install:
        try:
            subprocess.check_call(["pip", "install", package])
            print(f"已安装: {package}")
        except subprocess.CalledProcessError as e:
            print(f"安装失败: {package}")
            print(f"错误信息: {e}")

    print("请选择您的操作系统：")
    print("1: Windows 64位")
    print("2: Windows 32位")
    print("3: Linux")

    choice = input("请输入您的选择（1/2/3）：")

    # 获取当前工作目录
    current_dir = os.getcwd()
    # 构建openslide目录的路径
    openslide_dir = os.path.join(current_dir, 'openslide')

    # 根据用户的选择下载或输出指令
    if choice == '1':
        download_url = 'https://github.com/openslide/openslide-bin/releases/download/v20231011/openslide-win64-20231011.zip'
        download_file(download_url, openslide_dir)
    elif choice == '2':
        download_url = 'https://github.com/openslide/openslide-bin/releases/download/v20231011/openslide-win32-20231011.zip'
        download_file(download_url, openslide_dir)
    elif choice == '3':
        print("使用dnf install openslide-tools或apt install openslide-tools下载openslide环境！")
    else:
        print("无效的选择，请输入1、2或3。")

    # 获取当前工作目录
    current_dir = os.getcwd()
    # 构建openslide目录的路径
    openslide_dir = os.path.join(current_dir, 'openslide')
    find_and_unzip(openslide_dir)
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 构建openslide目录的路径
    openslide_dir = os.path.join(current_dir, 'openslide')
    move_contents_to_openslide(openslide_dir)
    print("操作完成。")

if __name__ == "__main__":
    main()
