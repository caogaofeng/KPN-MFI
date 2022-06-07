import torch
from torch.nn import functional as F

# guided filter
def guidedfilter(img, r, eps):
    img2 = torch.cat([img, img * img], dim=1)
    img2 = boxfilter(img2, r)

    m = torch.split(img2, 3, dim=1)
    mean_i, mean_ii = m[0], m[1]

    var_i = mean_ii - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    ab = torch.cat([a, b], dim=1)
    ab = boxfilter(ab, r)
    n = torch.split(ab, 3, dim=1)
    mean_a, mean_b = n[0], n[1]

    q = mean_a * img + mean_b

    return q


def boxfilter(x, szf):
    y = x
    szy = y.shape
    bf = torch.ones(szy[1], 1, szf, szf).float() / (szf ** 2)
    bf = bf.to(x)
    pp = int((szf - 1) / 2)
    y = F.pad(y, [pp, pp, pp, pp], mode='reflect')
    y = F.conv2d(y, bf, stride=1, groups=szy[1])
    return y
