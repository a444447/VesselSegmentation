from imgProcessed import  ImageProcessor
from getFeature import featureExtractor
import cv2
import numpy as np
from evalute import get_dice, saveImg
from PIL import Image

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('../data/raw/data/DRIVE/train/images/21_train.tif')
    # 使用Pillow读取GIF图像
    pil_img = Image.open('../data/raw/data/DRIVE/train/1st_manual/21_manual1.gif')

    # 将PIL图像转换为numpy数组
    ground = np.array(pil_img)

    _, g, _ = cv2.split(image)
    # 创建ImageChannelSperator对象
    processor = ImageProcessor(image=g)
    # 获得mask
    mask = processor.get_mask(image = g, thresh=30, maxval=255)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    kernel_size = (5, 5)
    upstream = processor.processData(kernel_size=kernel_size, sigma=0)

    # 创建featureExtractor对象
    extractor = featureExtractor(image=upstream)
    # 创建filter
    filters = extractor.create_filter()
    # 进行滤波
    gausssImg = extractor.process(filters)
    

    # 将mask应用到图像上
    maskedImg = extractor.apply_mask(mask=mask, image=gausssImg)
    grayStretchImg = extractor.gray_stretch(image=maskedImg)
    otsuImg = extractor.apply_otsu(thresh=30, maxval=255, image=grayStretchImg)
    predicted = otsuImg.copy()
    
    # 计算dice系数
    dice = get_dice(predicted, ground)
    print('dice:',dice)
    # 保存图像
    saveImg(image, ground, predicted)
    