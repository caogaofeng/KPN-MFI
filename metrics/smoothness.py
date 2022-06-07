import cv2
import numpy as np


def calculate_hdr_smoothness(LQ_img_path, GT_img_path):
    img1 = cv2.imread(LQ_img_path, -1)
    img2 = cv2.imread(GT_img_path, -1)
    # img1 = img1[:, :, [2, 1, 0]]
    # img2 = img2[:, :, [2, 1, 0]]
    img1, img2 = img1 / 65535.0, img2 / 65535.0
    hf_img1 = cv2.GaussianBlur(img1, (5, 5), 0.3)
    hf_img2 = cv2.GaussianBlur(img2, (5, 5), 0.3)
    return (np.sum(np.abs(img1 - hf_img1) ** 2), np.sum(np.abs(img2 - hf_img2) ** 2))
