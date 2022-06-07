import colour
import numpy as np
import cv2

def calculate_hdr_deltaITP(LQ_img_path, GT_img_path):
    img1 = cv2.imread(LQ_img_path, -1)
    img2 = cv2.imread(GT_img_path, -1)
    img1 = img1[:, :, [2, 1, 0]]
    img2 = img2[:, :, [2, 1, 0]]
    img1, img2 = img1/65535.0, img2/65535.0
    img1 = colour.models.eotf_ST2084(img1)
    img2 = colour.models.eotf_ST2084(img2)
    img1_ictcp = colour.RGB_to_ICTCP(img1)
    img2_ictcp = colour.RGB_to_ICTCP(img2)
    delta_ITP = 720 * np.sqrt((img1_ictcp[:, :, 0] - img2_ictcp[:, :, 0]) ** 2
                              + 0.25 * ((img1_ictcp[:, :, 1] - img2_ictcp[:, :, 1]) ** 2)
                              + (img1_ictcp[:, :, 2] - img2_ictcp[:, :, 2]) ** 2)
    return np.mean(delta_ITP)

