import cv2
import numpy as np


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr


def color_saturation(img, param):
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2bgr(ycbcr).astype(np.uint8)

    return img


def color_contrast(img, param):
    img = img.astype(np.float32) * param
    img = img.astype(np.uint8)

    return img


def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img


def jpeg_compression(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))
    return img


def get_distortion_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse
    # level starts from 1, list starts from 0
    return param_dict[type][level - 1]
