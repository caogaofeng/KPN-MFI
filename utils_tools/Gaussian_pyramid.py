import torch.nn as nn
import torch
import numpy as np
from utils_tools.guided_filter import guidedfilter


class GaussianBlur(nn.Module):
    def __init__(self, multi_channel=True):
        self.multi_channel = multi_channel
        super(GaussianBlur, self).__init__()
        kernel = np.array([[1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.],
                           [4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.],
                           [1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.]])

        kernel = torch.FloatTensor(kernel)
        if multi_channel:
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            self.gaussian = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3, bias=False)
        else:
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            self.gaussian = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.gaussian.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        x = self.gaussian(x)
        return x


class Gaussian_pyramid(nn.Module):
    def __init__(self, multi_channel=True, step=5):
        super(Gaussian_pyramid, self).__init__()
        self.Gau = GaussianBlur(multi_channel)
        self.step = step

    def forward(self, x):
        Gaussian_lists = [x]
        size_lists = [x.size()[2:]]
        for _ in range(self.step - 1):
            gaussian_down = self.Prdown(Gaussian_lists[-1])
            Gaussian_lists.append(gaussian_down)
            size_lists.append(gaussian_down.size()[2:])

        return Gaussian_lists

    def Prdown(self, x):
        x_ = self.Gau(x)
        x_ = x_[:, :, ::2, ::2]
        return x_

    def PrUp(self, x, sizes):
        b, c, _, _ = x.size()
        h, w = sizes
        up_x = torch.zeros((b, c, h, w), device='cuda')
        up_x[:, :, ::2, ::2] = x
        up_x = self.Gau_up(up_x)
        return up_x


class Gau_pyr_loss(nn.Module):
    def __init__(self, multi_channel=True, loss_mode='L1Loss', step=5):
        super(Gau_pyr_loss, self).__init__()
        self.gau = Gaussian_pyramid(multi_channel, step)
        if loss_mode == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_mode == 'L1Loss':
            self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0 / 2 ** (step - i) for i in range(1, step + 1)]
        # print(self.weights)

    def forward(self, x1L):

        cur = len(x1L) // 2
        x = guidedfilter(x1L[cur], r=11, eps=0.01)
        x_gau = self.gau(x)
        loss = 0
        for i in range(len(x1L)):
            if i == cur:
                continue
            else:
                x1 = guidedfilter(x1L[i], r=11, eps=0.01)
                x1_gau = self.gau(x1)
                for j in range(len(x1_gau)):
                    loss += self.weights[i] * self.criterion(x1_gau[j], x_gau[j])
        return loss


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gau_loss = Gau_pyr_loss(multi_channel=True).to(device)
    x = torch.rand(2, 3, 64, 64).to(device)
    y = torch.rand(2, 3, 64, 64).to(device)
    z = torch.rand(2, 3, 64, 64).to(device)

    r = gau_loss((x, y, z))

    print(r)
