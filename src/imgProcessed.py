'''
使用图像增强法来预处理图像，还需要先对其进行色彩空间变换和颜色通道的提取。
'''
import cv2
import os


class ImageProcessor:
    def __init__(self, image, save_path='./processed'):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.image = image


    def get_mask(self,thresh, maxval,image = None):
        """
        获取掩膜
        :param image: 输入的图像
        :param thresh: 阈值
        :param maxval: 最大值
        """
        if image is None:
            image = self.image
        _, mask = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
        return mask

    def equalizeHistProcessed(self, image=None):
        if image is None:
            image = self.image
            
        # 对G通道进行直方图均衡处理
        g_channel_equalized = cv2.equalizeHist(image)
        
        #cv2.imwrite(os.path.join(self.save_path, f"Equalized_G.png"), g_channel_equalized)
        return g_channel_equalized
    
    def GassuianProcessed(self,  kernel_size, sigma, image=None):
        """
        高斯滤波
        :param upstream: 输入的图像
        :param kernel_size: 高斯核大小
        :param sigma: 高斯核标准差
        """
        if image is None:
            image = self.image
        g_gaussian = cv2.GaussianBlur(image, kernel_size, sigma)
        #cv2.imwrite(os.path.join(self.save_path, f"Gaussian_G.png"), g_gaussian)
        return g_gaussian
    
    def processData(self, image = None, kernel_size=(5,5), sigma=0):
        if image is None:
            image = self.image
        # 对G通道进行直方图均衡处理

        gassuian_output = self.GassuianProcessed(image=image, kernel_size=kernel_size, sigma=sigma)
        g_channel_equalized = self.equalizeHistProcessed(image=gassuian_output)

       

        g_channel_clahe = self.clahe_processing(image=gassuian_output)
        #cv2.imwrite(os.path.join(self.save_path, f"Processed.png"), g_channel_clahe)
        return g_channel_clahe
    
    def clahe_processing(self, image=None):
        if image is None:
            image = self.image
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
        
        # 对G通道应用CLAHE处理
        g_channel_clahe = clahe.apply(image)
        
        #cv2.imwrite(os.path.join(self.save_path, f"CLAHE_G.png"), g_channel_clahe)
        return g_channel_clahe


#TODO    gamma校正


class ImageChannelSperator:
    def __init__(self, save_path='./channel_output'):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    

    def separate_channels(self, image, filename):
        """
        将图像分离为RGB和Lab的各个通道，并保存为灰度图像
        :param image: 输入的RGB图像
        :param filename: 图像的文件名，用于保存分离的通道
        """
        
        # RGB通道分离
        b_channel, g_channel, r_channel = cv2.split(image)
        cv2.imwrite(os.path.join(self.save_path, f"B_{filename}.png"), b_channel)
        cv2.imwrite(os.path.join(self.save_path, f"G_{filename}.png"), g_channel)
        cv2.imwrite(os.path.join(self.save_path, f"R_{filename}.png"), r_channel)

        # 将RGB图像转换为Lab色彩空间
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        # Lab通道分离
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        cv2.imwrite(os.path.join(self.save_path, f"L_{filename}.png"), l_channel)
        cv2.imwrite(os.path.join(self.save_path, f"A_{filename}.png"), a_channel)
        cv2.imwrite(os.path.join(self.save_path, f"B_lab_{filename}.png"), b_channel)