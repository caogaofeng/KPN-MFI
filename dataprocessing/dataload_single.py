import os
import random
import numpy as np
import torch
import torch.utils.data as data
import dataprocessing.util as util

def get_data_path_list(dir):
    path = [os.path.join(dir, i) for i in os.listdir(dir)]
    sorted(path)
    p = [path[3 * i:3 * (i + 1)] for i in range(len(path) // 3)]
    # print(len(p))
    list = []
    for i in range(len(p)):
        n = p[i]

        for j in range(len(n) - 1):
            temp = []
            temp.append(n[j])
            temp.append(n[j + 1])
            sorted(temp)
            list.append(temp)
    # print(len(list))
    return list

def get_data_path_list_2(dir):
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
            temp2.append(p2)
            temp2.append(p3)
            o.append(temp1)
            o.append(temp2)
        elif (p1 in org) and (p2 in org):
            temp1 = []
            temp1.append(p1)
            temp1.append(p2)
            o.append(temp1)
        elif (p2 in org) and (p3 in org):
            temp1 = []
            temp1.append(p2)
            temp1.append(p3)
            o.append(temp1)
        else:
            print(i)
            # os.remove(os.path.join(dir, i + '_3.png'))
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

        self.sizes_GT, self.paths_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.sizes_LQ, self.paths_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        # self.paths_LQ = get_data_path_list_2(opt['dataroot_LQ'])
        # self.paths_GT = get_data_path_list_2(opt['dataroot_GT'])


        assert self.paths_GT, 'Error: GT path is empty.'

        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(self.paths_GT), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):

        patch_size = self.opt['patch_size'] # crop size

        # get GT image
        GT_path = self.paths_GT[index // 100]
        # get LQ image
        LQ_path = self.paths_LQ[index // 100]

        img_LQ, img_GT = util.read_img(self.LQ_env, LQ_path), util.read_img(self.GT_env, GT_path)
        # img_LQ_1, img_LQ_2 = util.read_img(self.LQ_env, LQ_path[0]), util.read_img(self.LQ_env, LQ_path[1])

        if self.opt['phase'] == 'train':
            H, W, C = img_LQ.shape
            H_gt, W_gt, C = img_GT.shape
            assert H == H_gt, print('*******wrong image*******:{}'.format(LQ_path))

            # randomly crop
            if patch_size is not None:
                rnd_h = random.randint(0, max(0, H - patch_size))
                rnd_w = random.randint(0, max(0, W - patch_size))
                img_LQ = img_LQ[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
                img_GT = img_GT[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        mk = torch.from_numpy(np.ascontiguousarray(np.transpose(get_mask(img_LQ), (2, 0, 1)))).float()
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'MK': mk, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT) * 100


if __name__ == '__main__':
    # from options.options import Options
    #
    # opt = Options().parse()
    import yaml
    with open("../options/train/options.yml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    data = LQGT_dataset(opt)




