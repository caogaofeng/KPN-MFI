import numpy as np
import cv2
import os
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from utils_tools.E_itp import E_itp
from utils_tools.util import write_metrics_txt
import matlab.engine
from multiprocessing import Pool
from metrics.DeltaE_ITP import calculate_hdr_deltaITP
from metrics.smoothness import calculate_hdr_smoothness
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('base')
eng = matlab.engine.start_matlab()


def get_path_pair(path):
    gt_files = []
    lq_files = []
    for fname in sorted(os.listdir(path)):
        if fname.lower().startswith('gt_'):
            gt_files.append(os.path.join(path, fname))
        if fname.lower().startswith('lq_'):
            lq_files.append(os.path.join(path, fname))
    assert len(gt_files) == len(lq_files)
    sorted(gt_files)
    sorted(lq_files)
    return zip(lq_files, gt_files), len(gt_files)

def cal_psnr_ssim(lq_path, gt_path):
    lq = cv2.imread(lq_path, -1)
    gt = cv2.imread(gt_path, -1)
    lq, gt = lq / 65535.0 , gt/65535.0
    return compare_psnr(lq, gt), compare_ssim(lq, gt, multichannel=True)


def cal_matrics(lq, gt):
    vdp = eng.calculate_hdrvdp3(lq, gt)
    psnr, ssim = cal_psnr_ssim(lq, gt)
    eitp = calculate_hdr_deltaITP(lq, gt)
    srsim = eng.SR_SIM(lq, gt)
    # smooth = calculate_hdr_smoothness(lq, gt)
    logger.info('{:>10s} -  {:.6f} '.format('PSNR', psnr))
    logger.info('{:>10s} -  {:.6f} '.format('SSIM', ssim))
    logger.info('{:>10s} -  {:.6f} '.format('EITP', eitp))
    logger.info('{:>10s} -  {:.6f} '.format('SR_SIM', srsim))
    logger.info('{:>10s} -  {:.6f} '.format('HDR_VDP3', vdp))
    # logger.info('{:>10s} -  {:.6f}-{:.6f} '.format('smooth', smooth[0], smooth[1]))
    return vdp, psnr, ssim, eitp, srsim

def mul_cal_hdrvdp(n, pair, pair_len):
    pool = Pool(n)
    results = []
    for j in pair:
        assert str(j[0].split('\\')[-1][3:]).replace('SDR', 'HDR') == str(
            j[1].split('\\')[-1][3:]), 'gt and pred path not matching!!!!'
        s = pool.apply_async(cal_matrics, j)
        results.append(s)

    print()
    sum_results_vdp = 0
    sum_results_psnr = 0
    sum_results_ssim = 0
    sum_results_eitp = 0
    sum_results_srsim = 0
    sum_results_smooth_lq = 0
    sum_results_smooth_gt = 0
    for res in results:
        vdp, psnr, ssim, eitp, srsim = res.get()
        sum_results_vdp += vdp
        sum_results_psnr += psnr
        sum_results_ssim += ssim
        sum_results_eitp += eitp
        sum_results_srsim += srsim

    pool.close()
    pool.join()

    avg_hdr_vdp = sum_results_vdp / pair_len
    avg_psnr = sum_results_psnr / pair_len
    avg_ssim = sum_results_ssim / pair_len
    avg_eitp = sum_results_eitp / pair_len
    avg_srsim = sum_results_srsim / pair_len

    return avg_hdr_vdp, avg_psnr, avg_ssim, avg_eitp, avg_srsim

def get_data_path_list_2(dir, prex='LQ_', phase='test'):
    path = [i[:-6] for i in os.listdir(dir) if prex in i]
    org = [os.path.join(dir, i) for i in os.listdir(dir) if prex in i]
    s = list(set(path))
    s.sort()
    print(len(s))
    o = []
    for i in s:
        p1 = os.path.join(dir, i + '_1.png')
        p2 = os.path.join(dir, i + '_2.png')
        p3 = os.path.join(dir, i + '_3.png')
        if (p1 in org) and (p2 in org) and (p3 in org):
            temp1 = []
            temp2 = []
            temp1.append(p1)
            temp1.append(p2)
            if phase == 'train' and np.random.rand() < 0.5:
                temp1.reverse()
            temp2.append(p2)
            temp2.append(p3)
            if phase == 'train' and np.random.rand() < 0.5:
                temp2.reverse()
            o.append(temp1)
            o.append(temp2)
        elif (p1 in org) and (p2 in org):
            temp1 = []
            temp1.append(p1)
            temp1.append(p2)
            if phase == 'train' and np.random.rand() < 0.5:
                temp1.reverse()
            o.append(temp1)
        elif (p2 in org) and (p3 in org):
            temp1 = []
            temp1.append(p2)
            temp1.append(p3)
            if phase == 'train' and np.random.rand() < 0.5:
                temp1.reverse()
            o.append(temp1)
        else:
            print(i)
            # if os.path.exists(p1):
            #     os.remove(p1)
            # if os.path.exists(p2):
            #     os.remove(p2)
            # if os.path.exists(p3):
            #     os.remove(p3)
    # print(len(list))
    return o

def MABD(f1, f2):
    return np.mean(np.abs(f1 - f2))


def compute_MABD(path):
    path = get_data_path_list_2(path, prex='LQ_')
    print(len(path))
    result = 0
    for i in path:
        f1 = cv2.imread(i[0], -1)
        f2 = cv2.imread(i[1], -1)
        f1, f2 = f1 / 65535.0, f2 / 65535.0
        result += MABD(f1, f2)
    return result / len(path)

if __name__ == '__main__':
    path = '../test\evaluation/test_video_kpn_triple_v2'
    # path = 'F:\project\HDRTVNet-main\codes\experiments\highlight_generation/test_set'
    model_name = 'video_kpn_triple_v2'
    p, p_len = get_path_pair(path)
    print(p_len)

    avg_hdr_vdp, avg_psnr, avg_ssim, avg_eitp, avg_srsim = mul_cal_hdrvdp(8, p, p_len)
    mabd = compute_MABD(path)
    re = {'PSNR': avg_psnr,
          'SSIM': avg_ssim,
          'Eipt': avg_eitp,
          'HDR_VDP3': avg_hdr_vdp,
          'SR_SIM': avg_srsim,
          'MABD': mabd
          }

    print(re)
    write_metrics_txt('../test/test_result_metrics.txt', name=model_name, metrics=re)
