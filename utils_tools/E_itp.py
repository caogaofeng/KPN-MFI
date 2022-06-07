import numpy as np


def EOTF_inverse(F):
    # Y = F / 10000 #因为 F是归一化的，这个步骤略
    Y = F
    m1 = 2610/16384
    m2 = 2526 / 4096 * 128
    c2 = 18.8515625
    c3 = 18.6875
    c1 = c3 - c2 + 1

    return ((c1 + c2 * Y ** m1) / (1 + c3 * Y ** m1)) ** m2


def itp(im):
    R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    L = (1688 * R + 2146 * G + 262 * B) / 4096
    M = (683 * R + 2951 * G + 462 * B) / 4096
    S = (99 * R + 309 * G + 3688 * B) / 4096

    L_ = EOTF_inverse(L)
    M_ = EOTF_inverse(M)
    S_ = EOTF_inverse(S)

    I = 0.5 * L_ + 0.5 * M_
    CT = (6610 * L_ - 13613 * M_ + 7003 * S_) / 4096
    CP = (17933 * L_ - 17390 * M_ - 543 * S_) / 4096

    T = 0.5 * CT
    P = CP

    return I, T, P


def E_itp(im1, im2):
    I1, T1, P1 = itp(im1)
    I2, T2, P2 = itp(im2)
    return np.mean(720 * np.sqrt((I1 - I2) ** 2 + (T1 - T2) ** 2 + (P1 - P2) ** 2))
