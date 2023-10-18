import numpy as np
import cv2
from sklearn.model_selection import ParameterGrid
from train import train
params = {
    'gaussian_sigma': [1, 2, 3],  # 你可以填入实际的范围或值
    'gaussian_kernel_size': [(3,3), (5,5)],  # 同上
    'claheLimit': [1.0, 2.0],  # 同上
    'tileGridSize': [(8,8), (10,10)],  # 同上
    'gamma': [1.0, 1.5],  # 同上
    'L': [5, 10],  # 同上
    'sigma': [0.5, 1.0],  # 同上
    'm': [20.0/255, 30.0/255],  # 同上
    'E': [5, 8],  # 同上
}

def get_params():
    best_dice = 0
    best_params = {}
    param_grid = ParameterGrid(params)
    for param in param_grid:
        avg_dice = train(certain_param=param, train=True)
        print(f'Using params: {param} -> Dice score: {avg_dice:.4f}')

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_params = param
    print(f'Best Dice Score: {best_dice:.4f}')
    print(f'Best Parameters: {best_params}')
    return best_params, best_dice




def saveImg(src, ground, pred, dice_num):
    stackImg = np.hstack([src, cv2.cvtColor(ground,cv2.COLOR_GRAY2BGR),cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)])
    cv2.imwrite('../data/processed/' + 'structure:' + dice_num +'.png', stackImg)