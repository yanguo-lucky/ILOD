import cv2
import numpy as np


# 对图像进行单尺度Retinex处理
def single_scale_retinex(img, sigma):
    img = np.float64(img) + 1e-6
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


# 对图像进行多尺度Retinex处理
def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex


# 进行颜色恢复
def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True) + 1e-6
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


# 图像增强主函数，包括图像增强和颜色恢复
def retinex_process(img, sigma_list, G, b, alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_retinex = G * (img_retinex * img_color + b)

    # 将像素值限制在范围内
    for i in range(img_retinex.shape[2]):
        img_retinex[:, :, i] = np.clip(img_retinex[:, :, i], 0, 255)
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def main():
    # 读取图像
    img = cv2.imread('E:/pycharmprogect/2pcnet-master/datasets/bdd100k/val/b1c81faa-c80764c5.jpg')
    img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 尺度列表
    sigma_list = [15, 80, 150]
    # 增益参数
    G = 2.0
    # 偏置参数
    b = 10.0
    # 颜色恢复参数
    alpha = 100.0
    # 颜色恢复参数
    beta = 30.0

    # 进行图像增强
    img_retinex = retinex_process(img, sigma_list, G, b, alpha, beta)
    img_retinex = cv2.medianBlur(img_retinex, 3)

    # 显示原始图像
    cv2.imshow('1', img)
    # 显示增强后的图像
    cv2.imshow('Retinex', img_retinex)
    # 等待按键
    cv2.waitKey(0)
    # 保存增强后的图片
    cv2.imwrite('a.jpg', img_retinex)


if __name__ == "__main__":
    main()