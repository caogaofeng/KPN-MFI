import functools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch

# core1: pixel_convolution
def pixel_conv(feat, kernel):
    N, k2size, H, W = kernel.size()

    ksize = np.int(np.sqrt(k2size // 3))
    pad = (ksize - 1) // 2

    feat = F.pad(feat, [pad, pad, pad, pad])
    feat = feat.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat = feat.reshape(N, H, W, 3, -1)

    kernel = kernel.permute(0, 2, 3, 1).unsqueeze(-1)
    kernel = kernel.reshape(N, H, W, 3, -1)
    output = torch.mul(feat, kernel).sum(dim=-1, keepdim=True).squeeze(dim=-1)
    # output = output.reshape(N, H, W, -1)
    output = output.permute(0, 3, 1, 2).contiguous()
    return output

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class Guided_Attn(nn.Module):
    """ Guided attention Layer"""

    def __init__(self, in_dim=64):
        super(Guided_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, query, value):
        """
            inputs :
                x : input feature maps( B X C X W X H)
                guided_
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # att(query) * value + value
        # attention should be generated from query
        # and adopted the attention to value
        m_batchsize, C, width, height = value.size()

        proj_query = self.query_conv(query).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(query).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(value).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.gamma*out + x # residual add
        out = out + value  # residual add
        return out  # ,attention


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        # in_dim: input dims
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # out c/8
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # out c/8
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # out c
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # print(x.size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (C/8) X (N)

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X (C/8) x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.gamma*out + x # residual add
        out = out + x  # residual add
        return out  # ,attention

class SEModule(nn.Module):
    def __init__(self, num_channel_in, num_channel_out, squeeze_ratio=1.0):
        super(SEModule, self).__init__()
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)

        blocks = [nn.Linear(self.num_channel_in, int(self.num_channel_in * squeeze_ratio)),
                  nn.ReLU(),
                  nn.Linear(int(self.num_channel_in * squeeze_ratio), self.num_channel_out),
                  nn.Sigmoid()]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, fea):
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel_in)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel_out, 1, 1)
        x = fea * x
        return x


class encoder(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(encoder, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)

        self.HR_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.down_conv1 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(nf, nf, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(nf, nf, 3, 2, 1)

        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = make_layer(basic_block, 2)
        self.recon_trunk2 = make_layer(basic_block, 4)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.SA = Self_Attn(nf)
    def forward(self, x):

        fea0 = self.act(self.conv_first(x))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1 = self.recon_trunk1(fea1)

        fea2 = self.act(self.down_conv2(fea1))
        fea2 = self.recon_trunk2(fea2)

        fea3 = self.act(self.down_conv3(fea2))
        fea3 = self.SA(fea3)

        return [fea0, fea1, fea2, fea3]


class decoder(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(decoder, self).__init__()

        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.recon_trunk3 = make_layer(basic_block, 6)
        self.recon_trunk4 = make_layer(basic_block, 4)
        self.recon_trunk5 = make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv3 = nn.Sequential(nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2))

        self.HR_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Sequential(nn.Conv2d(nf, 5**2 * 3, 3, 1, 1, bias=True), nn.ReLU(inplace=True))

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x, fea):

        fea0, fea1, fea2, fea3 = fea[0], fea[1], fea[2], fea[3]

        out = self.recon_trunk3(fea3)
        out = fea3 + out

        out = self.act(self.up_conv1(out)) + fea2
        out = self.recon_trunk4(out)

        out = self.act(self.up_conv2(out)) + fea1
        out = self.recon_trunk5(out)

        out = self.act(self.up_conv3(out)) + fea0
        out = self.act(self.HR_conv2(out))

        core = self.conv_last(out)

        pred_h = pixel_conv(x, core)
        return pred_h


class MFI(nn.Module):
    def __init__(self, nf=64, n_frame=3):
        super(MFI, self).__init__()
        self.fusion = nn.Sequential(nn.Conv2d(2 * nf, nf, 3, 1, 1), nn.ReLU(True), nn.Conv2d(nf, nf, 1))
        for i in range(n_frame):
            name = 'SA_{:d}'.format(i + 1)
            setattr(self, name, Self_Attn(nf))
            if i + 1 < n_frame:
                name = 'GA_{:d}'.format(i + 1)
                setattr(self, name, Guided_Attn(nf))
        self.t_fus = nn.Conv2d((n_frame - 1) * nf, nf, 1)

    def forward(self, x):
        sa = []
        cur_id = len(x) // 2
        for i in range(len(x)):
            key = 'SA_{:d}'.format(i + 1)
            sa.append(getattr(self, key)(x[i]))
        mfi = []
        for i in range(cur_id):
            key = 'GA_{:d}'.format(i + 1)
            mfi.append(getattr(self, key)(sa[i], sa[cur_id]))
            key = 'GA_{:d}'.format(len(x) - i - 1)
            mfi.append(getattr(self, key)(sa[len(x) - i - 1], sa[cur_id]))

        assert len(mfi) + 1 == len(sa) == len(x), 'sequence length not odd'
        fusion = self.t_fus(torch.cat(mfi, dim=1))

        out = fusion * sa[cur_id] + sa[cur_id]
        return out


class HDRUnet_video(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, n_frame=3, act_type='relu'):
        super(HDRUnet_video, self).__init__()
        self.enc = encoder(in_nc=in_nc, out_nc=out_nc, nf=nf)
        self.dec = decoder(in_nc=in_nc, out_nc=out_nc, nf=nf)
        self.mfi = MFI(nf=nf, n_frame=n_frame)

    def mask3(self, img):
        # img --> N x C x H x W
        thr = 0.05
        msk = img - 1.0 + thr
        msk = torch.min(torch.ones_like(msk), (torch.max(torch.zeros_like(msk), msk) / thr))
        return msk

    def forward(self, x):  # x: (x1, x2, x3) tuple

        le = len(x)
        fea = []
        for i in range(le):
            # mk = self.mask3(x[i])
            f = self.enc(x[i])
            fea.append(f)
        H = []
        for i in range(le):
            idx = list_order(np.array(range(le)), i + 1).tolist()
            ffea = [fea[i][-1] for i in idx] # shuffle index
            mfi_f = self.mfi(ffea)
            fea[i][-1] = mfi_f
            predH = self.dec(x[i], fea[i])
            H.append(predH)
        return H


def list_order(x: np.array, id):
    x = x.copy()
    le = len(x)
    mid_id = le // 2

    if id - 1 <= mid_id:
        temp = x[id - 1]
        x[id - 1:mid_id] += 1
        x[mid_id] = temp
    if id > mid_id:
        temp = x[id - 1]
        x[mid_id:id] -= 1
        x[mid_id] = temp
    return x


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    x1 = torch.randn(2, 3, 64, 64)
    x2 = torch.randn(2, 3, 64, 64)
    x3 = torch.randn(2, 3, 64, 64)
    x4 = torch.randn(2, 3, 64, 64)
    m = HDRUnet_video(n_frame=3)

    with torch.no_grad():
        h = m((x, x1, x2 ))
        print(len(h))
        print(h[0].shape)
