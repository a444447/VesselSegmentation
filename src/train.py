from imgProcessed import  ImageProcessor
from getFeature import featureExtractor
import cv2
import numpy as np
from PIL import Image
import os
from scipy.signal import wiener


def get_dice(pred, ground):
    predict = pred.copy()
    groundtruth = ground.copy()
    predict[predict < 128] = 0
    predict[predict >= 128] = 1
    groundtruth[groundtruth < 128] = 0
    groundtruth[groundtruth >= 128] = 1
    predict_n = 1 - predict
    groundtruth_n = 1 - groundtruth
    TP = np.sum(predict * groundtruth)
    FP = np.sum(predict * groundtruth_n)
    TN = np.sum(predict_n * groundtruth_n)
    FN = np.sum(predict_n * groundtruth)
    # print(TP, FP, TN, FN)
    dice = 2 * np.sum(predict * groundtruth) / (np.sum(predict) + np.sum(groundtruth))
    return dice

def train(root='../data/raw/data/DRIVE/', train=False, certain_param=None):

    if not isinstance(certain_param, dict):
        ValueError('certain_param is not dict')
    if train:
        path = root + 'train/'
    else:
        path = root + 'test/'
    image_list = []

    kernel_size = certain_param['gaussian_kernel_size']
    sigma = certain_param['gaussian_sigma']
    clipLimit = certain_param['claheLimit']
    tileGridSize = certain_param['tileGridSize']
    gamma = certain_param['gamma']
    extractor_L = certain_param['L']
    extractor_sigma =  certain_param['sigma']
    m = certain_param['m']
    e = certain_param['E']

    total_dice = 0

    with os.scandir(path + 'images') as entries:
        image_list = [entry.name for entry in entries if entry.is_file()]
    for image in image_list:
        name, ext = image.split('.')
        index, _ = name.split('_')
        manual_path = path + '1st_manual/' + index + '_manual1.gif'

        # 读取图像
        image = cv2.imread(path + 'images/' + image)
        # 使用Pillow读取GIF图像
        pil_img = Image.open(manual_path)

        # 将PIL图像转换为numpy数组
        ground = np.array(pil_img)

        _, g, _ = cv2.split(image)
        # 创建ImageChannelSperator对象
        processor = ImageProcessor(image=g)
        # 获得mask
        mask = processor.get_mask(image = g, thresh=5, maxval=255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

        
        upstream = processor.processData(kernel_size=kernel_size, sigma=0, clipLimit=clipLimit, tileGridSize=tileGridSize, gamma=gamma)

        # 创建featureExtractor对象
        
        extractor = featureExtractor(image=upstream, L=extractor_L, sigma=extractor_sigma)
        # 创建filter
        filters = extractor.create_filter()
        # 进行滤波
        gausssImg = extractor.process(filters)
        

        # 将mask应用到图像上
        maskedImg = extractor.apply_mask(mask=mask, image=gausssImg)

       
        grayStretchImg = extractor.gray_stretch(image=maskedImg, m=m, e=e)
        otsuImg = extractor.apply_otsu(thresh=30, maxval=255, image=grayStretchImg) #thresh没有意义在otsu选项中
        predicted = otsuImg.copy()
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        # opened = cv2.morphologyEx(predicted, cv2.MORPH_OPEN, kernel)
        # denoised_image = opened.copy()
        # predicted = predicted.astype(float)
        # denoised_image = wiener(predicted, (3, 3))
        # denoised_image = np.clip(denoised_image, 0, 255).astype('uint8')
        # denoised_image = cv2.medianBlur(predicted, 3)
 
        # 计算dice系数
        dice = get_dice(predicted, ground)
        total_dice += dice
        # 保存图像
        #saveImg(image, ground, predicted, diceStr)
    return total_dice / len(image_list)