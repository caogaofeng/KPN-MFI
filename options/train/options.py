import argparse


class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # training options
        self.parser.add_argument('--batch_size', type=int, default=2, help='batch size for training network.')
        self.parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
        self.parser.add_argument('--models', type=str, default='video_kpn', help='choise models : Unet(default) or Unet_g')
        self.parser.add_argument('--patch_size', type=int, default=256, help='patch size')
        self.parser.add_argument('--phase', type=str, default='train', help='phase: train or test')
        self.parser.add_argument('--dataroot_GT', type=str, default='E:\\DATASET\\HDRTVNet\\video\\train_img_crop\\HDR', help='path of GT image')
        self.parser.add_argument('--dataroot_LQ', type=str, default='E:\\DATASET\\HDRTVNet\\video\\train_img_crop\\SDR', help='path of LQ image')

        self.parser.add_argument('--use_flip', type=bool, default=True, help='data augmentation with flip')
        self.parser.add_argument('--use_rot', type=bool, default=True, help='data augmentation with rot')
        self.parser.add_argument('--data_type', type=str, default='img', help='data type img or lmdb')

        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--lr_decay_after', type=int, default=10, help='linear decay of learning rate starts at this epoch')
        self.parser.add_argument('--continue_train', type=bool, default=True, help='continue training: load the latest models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # debugging options
        self.parser.add_argument('--save_ckpt_after', type=int, default=5, help='number of epochs after which checkpoints are saved')
        self.parser.add_argument('--save_img', type=int, default=5, help='number of epochs after which checkpoints are saved')
        self.parser.add_argument('--ckpt_image_path', type=str, default='./checkpoints/img', help='number of epochs after which checkpoints are saved')

        # testing options
        self.parser.add_argument('--ckpt_path', type=str, default='./checkpoints', help='path of checkpoint to be loaded')

    def parse(self):
        self.opt = self.parser.parse_args()
        print('========================================args========================================')
        for k in list(vars(self.opt).keys()):
            print('%s: %s' % (k, vars(self.opt)[k]))
        print('========================================args========================================\n')
        return self.opt
