import pdb

import torch
import numpy as np
from torchvision import transforms

from twophase.penet.danet import DENet


class ImagePreprocessor():
    def __init__(self):
        """初始化图像增强器，加载 DENet 模型"""
        self.model = DENet(num_high=3, ch_blocks=32, up_ksize=1, high_ch=32, high_ksize=3, ch_mask=32, gauss_kernel=5)

    def preprocess_image(self, image):
        """将输入的 NumPy 图像转换为 PyTorch 张量，并确保它是 float32 类型"""
        # 确保图像为 RGB 格式
        # print(f"image.shape: {image.shape}")
        if image.shape[0] == 3:  # 检查是否为 RGB 格式
            image_rgb = image
        else:
            raise ValueError("Input image must be in RGB format.")

        # 将图像调整为 PyTorch 张量
        if isinstance(image, torch.Tensor):
            # 如果是 uint8 类型，转换为 float32 类型
            if image.dtype == torch.uint8:
                image = image.float() / 255.0  # 归一化到 [0, 1] 范围
            else:
                image = image.float()  # 如果已经是 float 类型，直接转换为 float32
        else:
            raise ValueError("Input image must be a torch.Tensor.")

            # 增加批次维度
        image_tensor = image.unsqueeze(0)  # 增加批次维度，形状变为 [1, C, H, W]

        return image_tensor

    def postprocess_image(self, image_tensor):
        """将模型输出的 PyTorch 张量转换为张量格式（[0, 1] 范围）"""
        # 去掉批次维度并将张量从 [C, H, W] 格式转换为 [H, W, C] 格式
        image_tensor = image_tensor.squeeze(0).permute(0,1, 2)

        # 将像素值从 [0, 1] 范围转换到 [0, 255] 范围，并保证张量类型是 uint8
        image_tensor = (image_tensor * 255.0).clip(0, 255).to(torch.uint8)

        return image_tensor

    def enhance(self, image):
        """增强输入的图像并返回增强后的结果"""
        # 预处理图像

        input_tensor = self.preprocess_image(image)
        # print(f"enhance当中的image: {image}")
        # 将图像输入模型并获取增强后的结果
        with torch.no_grad():
            enhanced_tensor = self.model(input_tensor)

        # 后处理图像并返回
        enhanced_image = self.postprocess_image(enhanced_tensor)
        return enhanced_image

    def process_images_in_batch(self, image_list):
        """接受一个包含字典或元组的列表，提取图像并进行增强"""
        for item in image_list:
            # 提取每个元组或字典中的 'image' 键对应的值
            # print(f"process_images_in_batch的item: {item}")
            image = item['image'] if isinstance(item, dict) and 'image' in item else item[0]
            # print(f"process_images_in_batch的image: {image}")
            # 执行图像增强
            enhanced_image = self.enhance(image)

            # 替换原始的 'image' 键内容为增强后的图像
            item['image'] = enhanced_image
            # print(f"增强后的效果：{item['image'].shape}")

        return image_list
