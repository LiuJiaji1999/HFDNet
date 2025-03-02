import os
import numpy as np
import cv2  # 用于读取图像
from glob import glob

def get_image_paths(train_dir, val_dir, extensions=('jpg', 'png')):
    """ 获取所有图像的路径 """
    paths = []
    for ext in extensions:
        paths.extend(glob(os.path.join(train_dir, f'*.{ext}')))
        paths.extend(glob(os.path.join(val_dir, f'*.{ext}')))
    return paths

def calculate_mean_variance_image(image_paths):
    """ 增量法计算均值和方差图像 """
    count = 0  # 图像总数量
    sum_image = None  # 用于累积和
    sum_square_image = None  # 用于累积平方和

    for path in image_paths:
        image = cv2.imread(path).astype(np.float32) / 255.0  # 读取并归一化到 [0, 1]
        
        if sum_image is None:
            # 初始化累积和矩阵和平方和矩阵
            sum_image = np.zeros_like(image)
            sum_square_image = np.zeros_like(image)
        
        sum_image += image  # 累加和
        sum_square_image += image ** 2  # 累加平方和
        count += 1

    # 计算均值图像
    mean_image = sum_image / count
    
    # 计算方差图像
    variance_image = (sum_square_image / count) - (mean_image ** 2)
    
    # 计算标准差图像
    stddev_image = np.sqrt(variance_image)

    return mean_image, variance_image, stddev_image

# 设置train和val目录路径
train_dir = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/train'
val_dir = '/home/lenovo/data/liujiaji/DA-Datasets/CityScapes/yolov5_format/images/val'

# 获取所有图像路径
image_paths = get_image_paths(train_dir, val_dir)

# 计算均值图像和标准差图像
mean_image, variance_image, stddev_image = calculate_mean_variance_image(image_paths)

# 保存结果图像
cv2.imwrite('city-mean_image.png', (mean_image * 255).astype(np.uint8))
cv2.imwrite('city-variance_image.png', (variance_image * 255).astype(np.uint8))
cv2.imwrite('city-stddev_image.png', (stddev_image * 255).astype(np.uint8))

print("均值图像和标准差图像已保存！")

