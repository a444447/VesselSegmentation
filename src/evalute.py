import numpy as np
import cv2
from sklearn.model_selection import ParameterGrid
from train import train
params = {
    'gaussian_sigma': [1-0.2*i for i in range(5)],  # 你可以填入实际的范围或值
    'gaussian_kernel_size': [(3,3), (5,5), (7,7)],  # 同上
    'claheLimit': [limit for limit in range(2, 12, 2)],  # 同上
    'tileGridSize': [(8,8), (16,16), (32, 32)],  # 同上
    'gamma': [0.5 + i*0.5 for i in range(5)],  # 同上
    'L': [9.5, 10, 10.5],  # 同上
    'sigma': [1.0, 1.5],  # 同上
    'm': [i/255.0 for i in range(30, 250, 30) ],  # 同上
    'E': [2, 8, 10],  # 同上
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

def get_params_multi(param):
   avg_dice = train(certain_param=param, train=True)
   print(f'Using params: {param} -> Dice score: {avg_dice:.4f}')
   return avg_dice, param


def saveImg(src, ground, pred, dice_num):
    stackImg = np.hstack([src, cv2.cvtColor(ground,cv2.COLOR_GRAY2BGR),cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)])
    cv2.imwrite('../data/processed/' + 'structure:' + dice_num +'.png', stackImg)