import numpy as np
import cv2

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


def saveImg(src, ground, pred):
    stackImg = np.hstack([src, cv2.cvtColor(ground,cv2.COLOR_GRAY2BGR),cv2.cvtColor(pred,cv2.COLOR_GRAY2BGR)])
    cv2.imwrite('../data/processed/' + 'stack'+'.png', stackImg)