import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
import argparse
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
# from torch.optim import lr_scheduler
from dataprocessing.dataload_triple import LQGT_dataset
from torchvision.utils import save_image

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# from torch.nn.utils import clip_grad_norm
from models.KPN_MFI_video import HDRUnet_video
from utils_tools import util
from utils_tools.Gaussian_pyramid import Gau_pyr_loss


# from tensorboardX import SummaryWriter

# ========================================
#  training
# ========================================
def train(opt):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(util.dict2str(opt))
    # ======================================
    # loading data
    # ======================================
    dataset_train = LQGT_dataset(opt)
    train_data_loader = DataLoader(dataset_train, batch_size=opt['batch_size'],
                                   shuffle=True, num_workers=opt['num_workers'],
                                   # pin_memory=True,
                                   drop_last=True)

    print("Training samples: ", len(dataset_train))
    # ========================================
    # model init
    # ========================================
    model = HDRUnet_video(n_frame=3).cuda()
    # ========================================
    #  initialising losses and optimizer
    # ========================================
    l1 = torch.nn.L1Loss()
    # l2 = torch.nn.MSELoss()
    # vgg_loss = VGGLoss()
    con_loss = Gau_pyr_loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.9, 0.999))

    # learning rate scheduler here
    # scheduler = lr_scheduler.StepLR(optimizer_g, step_size=opt['lr_decay_after'], gamma=0.85)
    # ==================================================
    # loading checkpoints if continuing training
    # ==================================================
    if opt['continue_train']:
        try:
            loac_pt = './checkpoints/{}_latest.ckpt'.format(opt['model_name'])
            model.load_state_dict(torch.load(loac_pt))
            print('loading model from {} finish.'.format(loac_pt))
            start_epoch = 1
        except Exception as e:
            print(e)
            print('Checkpoint not found! Training from scratch.')
            start_epoch = 1
            # model.apply(weights_init)
    else:
        start_epoch = 1
        # model.apply(weights_init)
    # last_epoch_valid_loss = np.inf
    for epoch in range(start_epoch, opt['epochs'] + 1):
        running_loss = 0
        # loss_writer = SummaryWriter(log_dir='loss_graph')
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        ###########trainning############
        with tqdm(total=(len(dataset_train) - len(dataset_train) % opt['batch_size'])) as _tqdm:
            _tqdm.set_description('trian_epoch: {}/{}, lr: {:.2e}'.format(epoch, opt['epochs'], lr_cur[0]))
            for batch, data in enumerate(train_data_loader):

                LQ_1 = data['LQ_1'].float().cuda()
                GT_1 = data['GT_1'].float().cuda()
                LQ_2 = data['LQ_2'].float().cuda()
                GT_2 = data['GT_2'].float().cuda()
                LQ_3 = data['LQ_3'].float().cuda()
                GT_3 = data['GT_3'].float().cuda()
                path = data['LQ_path']

                # forward pass ->
                predH = model((LQ_1, LQ_2, LQ_3))
                content_loss = l1(torch.cat(predH), torch.cat([GT_1, GT_2, GT_3]))
                consistent_loss = con_loss(predH)
                Gloss = content_loss + 0.01 * consistent_loss

                optimizer.zero_grad()
                Gloss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
                optimizer.step()

                if (batch + 1) % 100 == 0:
                    save_image(
                        tensor=torch.cat((LQ_1, predH[0], GT_1, LQ_2, predH[1], GT_2, LQ_3,predH[2], GT_2), dim=0),
                        fp='./checkpoints/{}.jpg'.format(opt['model_name']),
                        nrow=LQ_1.shape[0]
                    )
                running_loss += Gloss.item()
                epoch_loss_g = running_loss / (batch + 1)
                _tqdm.set_postfix(g_loss='{:.5f}'.format(epoch_loss_g))
                _tqdm.update(len(LQ_1))

        # scheduler_g.step()
        if (epoch % opt['save_ckpt_after'] == 0):
            util.save_checkpoint(opt, epoch, model, epoch_loss_g)

    print('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../options/train/options_kpn_mfi_video.yml',
                        help='Path to config YMAL file.')
    params = parser.parse_args()
    if os.path.exists(params.config):
        with open(params.config, 'r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        train(opt)
    else:
        print('Please chaek your config yaml file')
