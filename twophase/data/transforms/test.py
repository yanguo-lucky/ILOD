import torch
import torchvision.transforms as T
import numpy as np
from numpy import random as R
import cv2


class NightAug:
    def __init__(self):
        self.gaussian = T.GaussianBlur(11, (0.1, 2.0))

    def mask_img(self, img, cln_img):
        while R.random() > 0.4:
            x1 = R.randint(img.shape[1])
            x2 = R.randint(img.shape[1])
            y1 = R.randint(img.shape[2])
            y2 = R.randint(img.shape[2])
            img[:, x1:x2, y1:y2] = cln_img[:, x1:x2, y1:y2]
        return img

    def gaussian_heatmap(self, x):
        """
        It produces a single Gaussian at a random point.
        """
        sig = torch.randint(low=1, high=150, size=(1,))[0]
        image_size = x.shape[1:]
        center = (torch.randint(image_size[0], (1,))[0], torch.randint(image_size[1], (1,))[0])
        x_axis = torch.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
        y_axis = torch.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
        xx, yy = torch.meshgrid(x_axis, y_axis)
        kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
        new_img = (x * (1 - kernel) + 255 * kernel).type(torch.uint8)
        return new_img

    def augment_image(self, img):
        img = torch.tensor(img).permute(2, 0, 1).cuda()  # HWC -> CHW, move to GPU
        g_b_flag = True

        # Gaussian Blur
        if R.random() > 0.5:
            img = self.gaussian(img)

        cln_img_zero = img.clone()

        # Gamma adjustment
        if R.random() > 0.5:
            cln_img = img.clone()
            val = 1 / (R.random() * 0.8 + 0.2)
            img = T.functional.adjust_gamma(img, val)
            img = self.mask_img(img, cln_img)
            g_b_flag = False

        # Brightness adjustment
        if R.random() > 0.5 or g_b_flag:
            cln_img = img.clone()
            val = R.random() * 0.8 + 0.2
            img = T.functional.adjust_brightness(img, val)
            img = self.mask_img(img, cln_img)

        # Contrast adjustment
        if R.random() > 0.5:
            cln_img = img.clone()
            val = R.random() * 0.8 + 0.2
            img = T.functional.adjust_contrast(img, val)
            img = self.mask_img(img, cln_img)
        img = self.mask_img(img, cln_img_zero)

        # Gaussian heatmap
        prob = 0.5
        while R.random() > prob:
            img = self.gaussian_heatmap(img)
            prob += 0.1

        # Add noise
        if R.random() > 0.5:
            n = torch.clamp(torch.normal(0, R.randint(50), img.shape), min=0).cuda()
            img = n + img
            img = torch.clamp(img, max=255).type(torch.uint8)

        return img.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC, move back to CPU


if __name__ == "__main__":
    image_path = "E:/pycharmprogect/2pcnet-master/datasets/bdd100k/val/b1c81faa-c80764c5.jpg"  # 替换为您的图像路径
    image = cv2.imread(image_path)

    if image is None:
        print("无法读取图像，请检查路径是否正确！")
    else:
        # 创建增强器实例
        augmenter = NightAug()

        # 增强图像
        augmented_image = augmenter.augment_image(image)

        # 可视化原图与增强后的图像
        cv2.imshow("Original Image", image)
        cv2.imshow("Augmented Image", augmented_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
