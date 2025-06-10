import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from twophase.penet.penet import PENet


# 1. 定义输入图片的转换函数
def load_image(image_path, size=(256, 256)):
    """加载并转换图片为适合PENet模型输入的格式"""
    img = Image.open(image_path).convert("RGB")  # 打开图片并转为RGB格式
    img = img.resize(size)  # 调整图片大小
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    img = transform(img)  # 应用转换
    img = img.unsqueeze(0)  # 添加batch维度
    return img


# 2. 定义输出图像的转换函数
def imshow(tensor_image):
    """显示tensor格式的图片"""
    tensor_image = tensor_image.squeeze(0).cpu().detach()  # 移除batch维度
    tensor_image = tensor_image.permute(1, 2, 0)  # 转换为HWC格式
    tensor_image = tensor_image * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])  # 反标准化
    tensor_image = tensor_image.numpy()  # 转换为numpy数组
    tensor_image = np.clip(tensor_image, 0, 1)  # 确保像素值在[0, 1]范围内
    plt.imshow(tensor_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()


# 3. 修改PENet类来输出增强效果
def enhance_image_with_PENet(image_path):
    """加载图像并通过PENet增强图像"""
    # 加载PENet模型
    model = PENet(num_high=3, ch_blocks=32, up_ksize=1, high_ch=32, high_ksize=3, ch_mask=32, gauss_kernel=5)
    model.eval()  # 设置为评估模式

    # 加载图像
    img = load_image(image_path)

    # 通过模型增强图像
    with torch.no_grad():  # 禁用梯度计算
        enhanced_img = model(img)  # 通过PENet处理图像

    # 显示增强后的图像
    imshow(enhanced_img)


# 4. 运行增强效果
image_path = "C:/Users/aplomb\Desktop/target.png"  # 这里替换成你自己的图片路径
enhance_image_with_PENet(image_path)
