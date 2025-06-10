import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from twophase.penet.danet import DENet


class ImagePreprocessor():
    def __init__(self):
        """初始化图像增强器，加载 DENet 模型"""
        self.model = DENet(num_high=3, ch_blocks=32, up_ksize=1, high_ch=32, high_ksize=3, ch_mask=32, gauss_kernel=5)
        self.model.eval()  # 设置为评估模式

    def preprocess_image(self, image):
        """将输入的图像路径或 NumPy 图像转换为 PyTorch 张量（如果输入已经是张量，则直接返回）"""

        # 如果输入是文件路径，读取图像并确保它是RGB格式
        if isinstance(image, str):  # 如果是文件路径
            image = Image.open(image).convert('RGB')

        # 将图像转换为 PyTorch 张量并增加批次维度
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 [C, H, W] 格式的张量
        ])
        image_tensor = transform(image).unsqueeze(0)  # 增加批次维度

        return image_tensor

    def postprocess_image(self, image_tensor):
        """将模型输出的 PyTorch 张量转换为 NumPy 图像"""
        # 去掉批次维度并将张量转换为 NumPy 格式
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # 将像素值从 [0, 1] 范围转换为 [0, 255]
        image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)

        return image_np

    def enhance(self, image):
        """增强输入的图像并返回增强后的结果"""
        # 预处理图像
        input_tensor = self.preprocess_image(image)

        # 将图像输入模型并获取增强后的结果
        with torch.no_grad():
            enhanced_tensor = self.model(input_tensor)

        # 后处理图像并返回
        enhanced_image = self.postprocess_image(enhanced_tensor)
        return enhanced_image

    def save_enhanced_image(self, image_path, enhanced_image, output_path):
        """保存增强后的图像"""
        enhanced_image_pil = Image.fromarray(enhanced_image)
        enhanced_image_pil.save(output_path)

    def process_and_show(self, image_path):
        """输入图像路径，输出增强后的图像，并展示结果"""
        # 增强图像
        enhanced_image = self.enhance(image_path)

        # 显示增强前后的图像
        original_image = Image.open(image_path).convert('RGB')

        # 使用Matplotlib显示图像
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(enhanced_image)
        ax[1].set_title('Enhanced Image')
        ax[1].axis('off')

        plt.show()

    def process_and_save(self, image_path, output_path):
        """输入图像路径，输出增强后的图像，并保存结果"""
        # 增强图像
        enhanced_image = self.enhance(image_path)

        # 保存增强后的图像
        self.save_enhanced_image(image_path, enhanced_image, output_path)


# 使用示例：

# 创建预处理器对象
preprocessor = ImagePreprocessor()

# 输入图像路径
image_path = 'C:/Users/aplomb/Desktop/target.png'

# 1. 显示增强前后的图像
preprocessor.process_and_show(image_path)

# 2. 保存增强后的图像
output_path = 'E:/pycharmprogect/2pcnet-master/output/test/1.jpg'
preprocessor.process_and_save(image_path, output_path)
