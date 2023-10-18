from evalute import get_params, params, get_params_multi
import time
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    path = '../data/raw/data/DRIVE/'
    print(params['m'])
    param_grid = ParameterGrid(params)
    with Pool() as pool:
        results = pool.map(get_params_multi, param_grid)
    best_dice, best_params = max(results, key=lambda x: x[0])

    end_time = time.time()  # 记录结束时间

    elapsed_time = end_time - start_time  # 计算运行时间
    
    print(f'Best Dice Score: {best_dice:.4f}')
    print(f'Best Parameters: {best_params}')
    print(f'total time:{elapsed_time:.4f}')
    # image_list = []
    # with os.scandir(path + 'images') as entries:
    #     image_list = [entry.name for entry in entries if entry.is_file()]
    # print(image_list)
    # for image in image_list:
    #     name, ext = image.split('.')
    #     index, _ = name.split('_')
    #     manual_path = path + '1st_manual/' + index + '_manual1.gif'

    #     # 读取图像
    #     image = cv2.imread(path + 'images/' + image)
    #     # 使用Pillow读取GIF图像
    #     pil_img = Image.open(manual_path)

    #     # 将PIL图像转换为numpy数组
    #     ground = np.array(pil_img)

    #     _, g, _ = cv2.split(image)
    #     # 创建ImageChannelSperator对象
    #     processor = ImageProcessor(image=g)
    #     # 获得mask
    #     mask = processor.get_mask(image = g, thresh=5, maxval=255)
    #     mask = cv2.erode(mask, np.ones((3, 3), np.uint8))

    #     kernel_size = (5, 5)
    #     sigma = 1
    #     clipLimit = 2.0
    #     tileGridSize = (10, 10)
    #     gamma = 1.5
    #     upstream = processor.processData(kernel_size=kernel_size, sigma=0, clipLimit=clipLimit, tileGridSize=tileGridSize, gamma=gamma)

    #     # 创建featureExtractor对象
    #     extractor_L = 10
    #     extractor_sigma = 1
    #     extractor = featureExtractor(image=upstream, L=extractor_L, sigma=extractor_sigma)
    #     # 创建filter
    #     filters = extractor.create_filter()
    #     # 进行滤波
    #     gausssImg = extractor.process(filters)
        

    #     # 将mask应用到图像上
    #     maskedImg = extractor.apply_mask(mask=mask, image=gausssImg)

    #     m = 30.0/255
    #     e = 8
    #     grayStretchImg = extractor.gray_stretch(image=maskedImg, m=m, e=e)
    #     otsuImg = extractor.apply_otsu(thresh=30, maxval=255, image=grayStretchImg) #thresh没有意义在otsu选项中
    #     predicted = otsuImg.copy()
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    #     # opened = cv2.morphologyEx(predicted, cv2.MORPH_OPEN, kernel)
    #     # denoised_image = opened.copy()
    #     # predicted = predicted.astype(float)
    #     # denoised_image = wiener(predicted, (3, 3))
    #     # denoised_image = np.clip(denoised_image, 0, 255).astype('uint8')
    #     # denoised_image = cv2.medianBlur(predicted, 3)
 
    #     # 计算dice系数
    #     dice = get_dice(predicted, ground)
    #     print('dice:',dice)
    #     diceStr = f'{dice:.4f}'
    #     # 保存图像
    #     saveImg(image, ground, predicted, diceStr)
    