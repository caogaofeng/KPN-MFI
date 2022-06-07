import torch, cv2, os
import numpy as np
from numpy.random import uniform
from random import randint
import torch.nn as nn
import math


def load_checkpoint(model, opt):
    """ loading checkpoints for continuing training or evaluation """
    path = os.path.join(opt.ckpt_path, opt.model + '_' + 'latest.ckpt')
    print("load checkpoints from:", path)
    start_epoch = np.loadtxt(os.path.join(opt.ckpt_path, opt.model + '_' + 'state.txt'), dtype=int)
    model.load_state_dict(torch.load(path))
    print('Resuming from epoch ', start_epoch)
    return start_epoch, model


def make_required_directories(mode):
    if mode == 'train':
        if not os.path.exists('./checkpoints'):
            print('Making checkpoints directory')
            os.makedirs('./checkpoints')

        if not os.path.exists('./test_results_4'):
            print('Making test_results_4 directory')
            os.makedirs('./test_results_4')
    elif mode == 'test_HDREye':
        if not os.path.exists('./test_results'):
            print('Making test_results directory')
            os.makedirs('./test_results')


def mu_tonemap(img, mu=5000):
    MU = mu
    # re = torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)
    re = torch.log(1.0 + MU * img) / np.log(1.0 + MU)
    return re


def save_checkpoint(opt, epoch, model, loss):
    """ Saving model checkpoint """
    checkpoint_path = os.path.join(opt['ckpt_path'],
                                   opt['model_name'] + '_' + 'epoch_' + str(epoch) + "_loss_" + str(loss)[:7] + '.ckpt')
    latest_path = os.path.join(opt['ckpt_path'], opt['model_name'] + '_' + 'latest.ckpt')
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    np.savetxt(os.path.join(opt['ckpt_path'], opt['model_name'] + '_' + 'state.txt'), [epoch + 1], fmt='%d')
    print('Saved checkpoint for epoch ', epoch)


def update_lr(optimizer, epoch, opt):
    """ Linearly decaying model learning rate after specified (opt.lr_decay_after) epochs """
    new_lr = opt.lr - opt.lr * (epoch - opt.lr_decay_after) / (opt.epochs - opt.lr_decay_after)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Learning rate decayed. Updated LR is: %.6f' % new_lr)


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def CV_TMO(Eim, flag=1):
    if int(flag) == 1:
        gamma = uniform(1.8, 2.2)
        intensity = uniform(-1.0, 1.0)
        light_adapt = uniform(0.8, 1.0)
        color_adapt = uniform(0.0, 0.2)
        # tonemap = cv2.createTonemapReinhard(gamma=gamma, intensity=intensity, light_adapt=light_adapt,
        #                                     color_adapt=color_adapt)
        tonemap = cv2.createTonemapReinhard(gamma=2.0)
    elif int(flag) == 2:
        gamma = uniform(1.8, 2.2)
        scale = uniform(0.65, 0.85)
        tonemap = cv2.createTonemapMantiuk(saturation=1.0, scale=scale, gamma=gamma)
        # tonemap = cv2.createTonemapMantiuk()
    elif int(flag) == 3:
        gamma = uniform(1.8, 2.2)
        bias = uniform(0.7, 0.9)
        tonemap = cv2.createTonemapDrago(saturation=1.0, bias=bias, gamma=gamma)
        # tonemap = cv2.createTonemapDrago()

    ldrReinhard = tonemap.process(Eim)

    return ldrReinhard


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def random_tone_map(x):
    x = map_range(x)
    choice = np.random.randint(0, 3)
    # print(choice + 1)
    tmo = CV_TMO(x, 1)
    # print(tmo.min(), tmo.max())
    return map_range(tmo)


def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""
    img = np.zeros((height, width, channels), np.float32)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    return 1 - img


def get_mask(ldr, percent=0.95):
    m = np.clip(ldr - percent, 0, None) / (1 - percent)
    return np.clip(m, 0, 1)


def ToTensor(img):
    img_chw = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_chw.copy()).float()
    return img_tensor


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name, map_location='cpu')
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])

    epoch = ckpt_dict['n_iter'] if 'n_iter' in ckpt_dict else 0
    step = ckpt_dict['step'] if 'step' in ckpt_dict else 0

    return step, epoch


def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0: r + 1, :] = imCum[r: 2 * r + 1, :]
    imDst[r + 1: rows - r, :] = imCum[2 * r + 1: rows, :] - imCum[0: rows - 2 * r - 1, :]
    imDst[rows - r: rows, :] = np.tile(imCum[rows - 1, :], [r, 1]) - imCum[rows - 2 * r - 1: rows - r - 1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
    imDst[:, r + 1: cols - r] = imCum[:, 2 * r + 1: cols] - imCum[:, 0: cols - 2 * r - 1]
    imDst[:, cols - r: cols] = np.tile(imCum[:, cols - 1], [r, 1]).T - imCum[:, cols - 2 * r - 1: cols - r - 1]

    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols, c) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''Converts a torch Tensor into an image Numpy array'''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)  # np.float64
    img2 = img2.astype(np.float32)  # np.float64
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def mask3(self, img):
    # img --> N x C x H x W
    thr = 0.05
    msk = img - 1.0 + thr
    msk = torch.min(torch.ones_like(msk), (torch.max(torch.zeros_like(msk), msk) / thr) ** 2)
    return msk


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def cam_w_batch(im_tensor, size=256):
    b, c, h, w = im_tensor.shape
    im = im_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    result = []
    for i in range(b):
        x = im[i]
        x = x - np.min(x)
        cam_img = x / np.max(x)
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, (size, size))
        cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        cam_img = cam_img / 255.0
        result.append(torch.from_numpy(cam_img).permute(2, 1, 0).unsqueeze(0))
    return torch.cat(result, dim=0)


def write_metrics_txt(path, name, metrics: dict):
    with open(path, 'a', encoding='ascii') as f:
        f.write(str(name))
        f.write('\n')
        for k, v in metrics.items():
            f.write('{:>10s}:'.format(k))
            f.write('{:.6f}'.format(v))
            f.write('\n')
        f.write('\n')
        f.close()


if __name__ == '__main__':
    hdr_path = "E:\\DATASET\\HDRTVNet\\video\\YouTebe_video\\test_frame_resize_down_x5\\HDR\\HD_Club_HDR_Demo_HDR10_HDR_9.00_1.png"
    # ldr_path = "E:\DATASET\\NTIRE2021_HDR\\train\Train0000_0210\\0000_medium.png"
    ldr_path = "E:\\DATASET\HDRTVNet\\video\\YouTebe_video\\test_frame_resize_down_x5\\SDR\\HD_Club_HDR_Demo_HDR10_SDR_9.00_1.png"

    hdr = cv2.imread(hdr_path, -1).astype(np.float32)
    # ldr = cv2.imread(ldr_path, -1).astype(np.float32)
    ldr = cv2.imread(ldr_path).astype(np.float32)
    hdr = hdr / 65535.0
    ldr = ldr / 255.0

    # cv2.ximgproc.guidedFilter 引导滤波器
    # cv2.bilateralFilter()
    # hdr_base = cv2.ximgproc.guidedFilter(hdr, hdr, 5, 0.1)
    # ldr_base = cv2.ximgproc.guidedFilter(ldr, ldr, 5, 0.1)
    hdr_base = cv2.bilateralFilter(hdr, 9, 75, 75)
    ldr_base = cv2.bilateralFilter(ldr, 9, 75, 75)
    hdr_detail = hdr / hdr_base
    ldr_detail = ldr / ldr_base

    cv2.imshow('hdr_base', (hdr_base * 65535).astype(np.uint16))
    cv2.imshow('ldr_base', (ldr_base * 255).astype(np.uint8))
    cv2.imshow('hdr_detail', (hdr_detail * 65535).astype(np.uint16))
    cv2.imshow('ldr_detail', (ldr_detail * 255).astype(np.uint8))
    cv2.waitKey(0)


