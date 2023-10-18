import numpy as np
import cv2
import os
class featureExtractor:
    def __init__(self, L=10, sigma = 1, image=None):
        self.L = L
        self.sigma = sigma
        if image is None:
            raise ValueError('image is None')
        self.image = image

    def create_filter(self):
        filters = []
        #计算filter的宽度
        kernel_size = np.ceil(np.sqrt((6 * np.ceil(self.sigma) + 1) ** 2 + self.L ** 2))
        #保证kernel_size是奇数，以便找得到中心点
        if np.mod(kernel_size, 2) == 0:
            kernel_size = kernel_size + 1
        kernel_size = int(kernel_size)

        #计算每个filter的角度(0-180,每次15度)
        for theta in np.arange(0, np.pi, np.pi / 16):
            match_filter = np.zeros((kernel_size, kernel_size), dtype=np.float)
            for x in range(kernel_size):
                for y in range(kernel_size):
                    radius = (kernel_size - 1) / 2
                    #下面使用旋转矩阵坐标变换，将(x,y)转换为(x_,y_)，这样可以为滤波器添加方向性
                    x_ = (x - radius) * np.cos(theta) + (y - radius) * np.sin(theta)
                    y_ = -(x - radius) * np.sin(theta) + (y - radius) * np.cos(theta)
                    #对于在高斯曲线以外的像素，或者长度超过L的像素，将其置为0
                    if abs(x_) > 3 * np.ceil(self.sigma):
                        match_filter[x, y] = 0
                    elif abs(y_) > (self.L - 1) / 2:
                        match_filter[x, y] = 0
                    else:
                        #计算高斯曲线的值， 只计算了当前theta下，x方法的部分。因为y方向的部分已经进行了条件限制
                        match_filter[x, y] = -np.exp(-.5 * (x_ / self.sigma) ** 2) / (np.sqrt(2 * np.pi) * self.sigma)
            m = 0.0
            for i in range(match_filter.shape[0]):
                for j in range(match_filter.shape[1]):
                    if match_filter[i, j] < 0:
                        m = m + 1
            mean = np.sum(match_filter) / m
            for i in range(match_filter.shape[0]):
                for j in range(match_filter.shape[1]):
                    if match_filter[i, j] < 0:
                        match_filter[i, j] = match_filter[i, j] - mean
            filters.append(match_filter)
        return filters
    
    def process(self, filters):
        res = np.zeros_like(self.image)
        for kern in filters:
            fimg = cv2.filter2D(self.image, cv2.CV_8U, kern, borderType=cv2.BORDER_REPLICATE)
            res = np.maximum(res, fimg)
        #cv2.imwrite(os.path.join('./processed', f"CLAHE_G.png"), res)
        return res
    
    def apply_mask(self, mask, image=None):
        if image is None:
            image = self.image
        masked_image = image.copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 0:
                    masked_image[i][j] = 0
        return masked_image
    
    def gray_stretch(self,m=30.0/255, e = 8, image=None):

        if image is None:
            image = self.image

        k = np.zeros(image.shape, np.float)
        ans = np.zeros(image.shape, np.float)
        mx = np.max(image)
        mn = np.min(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                k[i][j] = (float(image[i][j]) - mn) / (mx - mn)
        eps = 0.01
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ans[i][j] = 1 / (1 + (m / (k[i][j] + eps)) ** e) * 255
        ans = np.array(ans, np.uint8)
        return ans
    
    def apply_otsu(self, thresh, maxval, image=None):
        if image is None:
            image = self.image
        _, th1 = cv2.threshold(image, thresh, maxval, cv2.THRESH_OTSU)
        return th1
    
    