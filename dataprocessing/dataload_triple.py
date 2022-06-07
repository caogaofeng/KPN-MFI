import os
import random
import numpy as np
import torch
import torch.utils.data as data
import dataprocessing.util as util


def get_data_path_list_2(dir, phase='train'):
    path = [i[:-6] for i in os.listdir(dir)]
    org = [os.path.join(dir, i) for i in os.listdir(dir)]
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

def get_data_path_list_3(dir, phase='train'):
    path = [i[:-6] for i in os.listdir(dir)]
    org = [os.path.join(dir, i) for i in os.listdir(dir)]
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
            temp1.append(p1)
            temp1.append(p2)
            temp1.append(p3)
            if phase == 'train' and np.random.rand() < 0.5:
                random.shuffle(temp1)
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



def get_mask(ldr, percent=0.95):
    m = np.clip(ldr - percent, 0, None) / (1 - percent)
    return np.clip(m, 0, 1)


class LQGT_dataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGT_dataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        # self.paths_LQ = get_data_path_list(opt['dataroot_LQ'], opt['phase'])
        self.paths_LQ = get_data_path_list_3(opt['dataroot_LQ'], opt['phase'])
        # self.paths_LQ.sort()
        # self.paths_GT.sort()
        assert self.paths_LQ, 'Error: LQ path is empty.'

        # if self.paths_LQ and self.paths_GT:
        #     assert len(self.paths_LQ) == len(self.paths_GT), 'GT and LQ datasets have different number of images - {}, {}.'.format(
        #         len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):

        patch_size = self.opt['patch_size']  # crop size

        # get LQ image
        LQ_path = self.paths_LQ[index]
        # get GT image
        GT_path = [p.replace('SDR', 'HDR') for p in LQ_path]
        # assert os.path.basename(LQ_path[1]).replace('SDR', 'HDR') == os.path.basename(GT_path[1]), 'image name not equal!'
        # assert os.path.basename(LQ_path[0]).replace('SDR', 'HDR') == os.path.basename(GT_path[0]), 'image name not equal!'

        img_GT_1, img_GT_2,img_GT_3 = util.read_img(self.GT_env, GT_path[0]), \
                                      util.read_img(self.GT_env, GT_path[1]),\
                                      util.read_img(self.GT_env, GT_path[2])
        img_LQ_1, img_LQ_2, img_LQ_3 = util.read_img(self.LQ_env, LQ_path[0]), \
                                       util.read_img(self.LQ_env, LQ_path[1]), \
                                       util.read_img(self.LQ_env, LQ_path[2])

        if self.opt['phase'] == 'train':
            H, W, C = img_LQ_1.shape
            H_gt, W_gt, C = img_GT_1.shape
            assert H == H_gt, print('*******wrong image*******:{}'.format(LQ_path))

            # randomly crop
            if patch_size is not None:
                rnd_h = random.randint(0, max(0, H - patch_size))
                rnd_w = random.randint(0, max(0, W - patch_size))
                img_LQ_1 = img_LQ_1[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_GT_1 = img_GT_1[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_LQ_2 = img_LQ_2[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_GT_2 = img_GT_2[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_LQ_3 = img_LQ_3[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_GT_3 = img_GT_3[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

            # augmentation - flip, rotate
            img_LQ_1, img_GT_1, img_LQ_2, img_GT_2, img_LQ_3, img_GT_3 = util.augment([img_LQ_1, img_GT_1, img_LQ_2, img_GT_2, img_LQ_3, img_GT_3],
                                                                  self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT_1.shape[2] == 3:
            img_GT_1 = img_GT_1[:, :, [2, 1, 0]]
            img_LQ_1 = img_LQ_1[:, :, [2, 1, 0]]
            img_GT_2 = img_GT_2[:, :, [2, 1, 0]]
            img_LQ_2 = img_LQ_2[:, :, [2, 1, 0]]
            img_GT_3 = img_GT_3[:, :, [2, 1, 0]]
            img_LQ_3 = img_LQ_3[:, :, [2, 1, 0]]

        mk_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(get_mask(img_LQ_1), (2, 0, 1)))).float()
        mk_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(get_mask(img_LQ_2), (2, 0, 1)))).float()
        mk_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(get_mask(img_LQ_3), (2, 0, 1)))).float()
        img_GT_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_1, (2, 0, 1)))).float()
        img_LQ_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_1, (2, 0, 1)))).float()

        img_GT_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_2, (2, 0, 1)))).float()
        img_LQ_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_2, (2, 0, 1)))).float()

        img_GT_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_3, (2, 0, 1)))).float()
        img_LQ_3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_3, (2, 0, 1)))).float()

        L_s = np.array([0.1])[np.newaxis, np.newaxis, :]
        H_s = np.array([0.9])[np.newaxis, np.newaxis, :]

        return {'LQ_1': img_LQ_1, 'MK_1': mk_1, 'GT_1': img_GT_1,
                'LQ_2': img_LQ_2, 'MK_2': mk_2, 'GT_2': img_GT_2,
                'LQ_3': img_LQ_3, 'MK_3': mk_3, 'GT_3': img_GT_3,
                'LQ_path': LQ_path, 'GT_path': GT_path,
                'L_s': L_s, 'H_s': H_s
                }

    def __len__(self):
        return len(self.paths_LQ)


if __name__ == '__main__':
    # from options.options import Options
    #
    # opt = Options().parse()
    import yaml

    with open("../options/train/options.yml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    data = LQGT_dataset(opt)
