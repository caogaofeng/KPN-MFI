import numpy as np
import cv2
import os
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from utils_tools.E_itp import E_itp
from utils_tools.util import write_metrics_txt

def get_path_pair(path):
    gt_files = []
    gd_files = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().startswith('gt_'):
            gt_files.append(os.path.join(path, fname))
        if fname.lower().startswith('lq_'):
            gd_files.append(os.path.join(path, fname))
    assert len(gt_files) == len(gd_files)
    sorted(gt_files)
    sorted(gd_files)
    return zip(gt_files, gd_files), len(gt_files)


if __name__ == '__main__':
    # path = '../test\evaluation/test_vedio_kpn_consistence'
    path = 'F:\project\HDRTVNet-main\codes\experiments\highlight_generation/test_set'
    model_name = 'HG'
    p, p_len = get_path_pair(path)
    print(p_len)
    PSNR = []
    SSIM = []
    Eitp = []
    for gt, pred in p:
        print(gt, pred)
        assert str(pred.split('\\')[-1][3:]).replace('SDR', 'HDR') == str(gt.split('\\')[-1][3:]), 'gt and pred path not matching!!!!'
        gt_im = cv2.imread(gt, -1)
        pred_im = cv2.imread(pred, -1)

        ## BGR2RGB
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)
        pred_im = cv2.cvtColor(pred_im, cv2.COLOR_BGR2RGB)

        ## normalizeing
        gt_im = gt_im / 65535.0
        pred_im = pred_im / 65535.0

        PSNR.append(compare_psnr(pred_im, gt_im))
        SSIM.append(compare_ssim(pred_im, gt_im, multichannel=True))
        Eitp.append(E_itp(pred_im, gt_im))

    re = {'PSNR': sum(PSNR)/len(PSNR),
          'SSIM': sum(SSIM) / len(SSIM),
          'Eipt': sum(Eitp) / len(Eitp)
          }

    print(re)
    write_metrics_txt('../test/test_result_metrics.txt', name=model_name, metrics=re)
