# coding=utf-8

import cv2
import albumentations as A
import os
import numpy as np
from datetime import datetime
import random

import albumentations as A

# 定义每张原图要生成几张增强后图片
num_aug = 5

# 定义增强流水线
transform = A.Compose([
    # 几何变换
    A.RandomRotate90(p=0.5),  # 随机旋转90度
    A.HorizontalFlip(p=0.5),  # 随机水平或垂直翻转
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),  # 平移、缩放和旋转

    # 颜色变换
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 随机调整亮度和对比度
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # 随机调整色调、饱和度和亮度

    # 分割和裁剪
    # A.RandomCrop(height=256, width=256, p=0.3),  # 随机裁剪
    # A.CenterCrop(height=256, width=256, p=0.3),  # 中心裁剪

    # 特殊效果
    A.RandomShadow(p=0.2),  # 随机添加阴影

    # 噪声
    A.GaussNoise(p=0.2),  # 随机添加高斯噪声

    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),  # 随机选择运动模糊、中值模糊或普通模糊

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 输入图片目录
input_image_directory = r'D:\yolo_test\1-标注工具\test_flies\flies\train'

# 输入标注文件目录
input_label_directory = r'D:\yolo_test\1-标注工具\test_flies\flies\labels'

# 输出目录
# output_directory = r'D:\4-college\course\PycharmProjects\yolov10\augmented'
output_image_directory = r'D:\yolo_test\2-数据增强\augmented\i2'

output_label_directory = r'D:\yolo_test\2-数据增强\augmented\l2'

# 确保输出目录存在
os.makedirs(output_image_directory, exist_ok=True)
os.makedirs(output_label_directory, exist_ok=True)

# 获取目录中的所有图片文件
image_files = [f for f in os.listdir(input_image_directory) if f.endswith('.jpg')]

# 遍历所有图片文件
for image_file in image_files:
    # 输入图片路径
    input_image_path = os.path.join(input_image_directory, image_file)

    # 输入标注文件路径
    input_label_path = os.path.join(input_label_directory, image_file.replace('.jpg', '.txt'))
    print(input_label_path)

    # 读取图片
    # image = cv2.imread(input_image_path)
    image = cv2.imdecode(np.fromfile(input_image_path), cv2.IMREAD_COLOR)  # 如有中文路径，需使用imencode方法
    if image is None:
        print(f"Failed to read image: {input_image_path}")
        continue
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #转为RGB, 目前 albumentations 已不需要做这一步。

    # 读取标注文件
    if not os.path.exists(input_label_path):
        print(f"Label file not found: {input_label_path}")
        continue

    with open(input_label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = list(map(float, parts[1:]))
        bboxes.append(bbox)
        class_labels.append(class_id)

    # 生成num_aug张增强后的图片和标注文件
    for i in range(num_aug):
        # 应用增强变换
        # random.seed(42)
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']

        # 输出图片路径
        output_image_path = os.path.join(output_image_directory,
                                         f'augmented_{os.path.splitext(image_file)[0]}_{i + 1}.jpg')

        # 保存增强后的图片
        # cv2.imwrite(output_image_path, augmented_image)
        cv2.imencode('.png', augmented_image)[1].tofile(output_image_path)  # 如有中文路径，需使用imencode方法

        # 输出标注文件路径
        output_label_path = os.path.join(output_label_directory,
                                         f'augmented_{os.path.splitext(image_file)[0]}_{i + 1}.txt')

        # 保存增强后的标注文件
        with open(output_label_path, 'w') as f:
            for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
                bbox_str = ' '.join(map(str, bbox))
                f.write(f'{class_id} {bbox_str}\n')

        print(
            f"Augmented image and label {i + 1} for {image_file} saved to {output_image_path} and {output_label_path}")