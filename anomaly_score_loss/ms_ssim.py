# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class SSIM_Loss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:

        """Computes the structural similarity (SSIM) index map between two images
        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)

    def forward(self, x: Tensor, y: Tensor):

        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)

        l_map, cs_map = self._ssim(x, y)
        return l_map, cs_map

    def _ssim(self, x: Tensor, y: Tensor):

        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)

        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=3)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        cs_map = (2 * vxy + c2) / (vx + vy + c2)  # set alpha=beta=gamma=1
        l_map = ((2 * ux * uy + c1) / (ux ** 2 + uy ** 2 + c1))  # 亮度项
        return l_map, cs_map

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:

        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d

# multi-scale structural similarity loss
class MSSSIM_Loss(torch.nn.Module):
    def __init__(self, num_scales=4):
        super(MSSSIM_Loss, self).__init__()
        self.num_scales = num_scales
        self.model = SSIM_Loss()

    def forward(self, img_1, img_2, as_loss=False): # img_1:target img_2:input
        b, c, h, w = img_1.shape

        mcs = []
        l = None
        for scale in range(self.num_scales):
            if scale > 0:
                img_1 = F.avg_pool2d(img_1, kernel_size=2, stride=2, padding=0)
                img_2 = F.avg_pool2d(img_2, kernel_size=2, stride=2, padding=0)

            l_map, cs_map = self.model(img_1, img_2)
            if l is None:
                l = l_map
            mcs.append(F.interpolate(cs_map, size=(h, w), mode="bilinear", align_corners=False))

        msssim_map = torch.mean(torch.stack(mcs), dim=0) * l
        if as_loss:
            return torch.mean(1 - msssim_map)
        else:
            return torch.mean(1 - msssim_map, axis=1).unsqueeze(1)


# if __name__ == '__main__':
#     from dataloader import load_data
#     from main import parse_arg
#     from torchvision.utils import save_image
#     from utils import unorm_image

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)

#     opt = parse_arg()
#     data_loader = load_data(opt)

#     for iteration, (datas, masks, labels) in enumerate(data_loader.valid):
#         if iteration == 0:
#             img_1 = datas.to(device)
#         elif iteration == 1:
#             img_2 = datas.to(device)
#             break

#     model = MSSSIM()
#     map = model(img_1, img_2, as_loss=False)
#     print(map)
#     print(map.shape)
#     print(torch.min(map))
#     print(torch.max(map))
#     save_image(unorm_image(img_1, opt.data_mean, opt.data_std), '1.png', nrow=3)
#     save_image(unorm_image(img_2, opt.data_mean, opt.data_std), '2.png', nrow=3)
#     save_image(map, '3.png', nrow=3)

#     loss = model(img_1, img_2, as_loss=True)
#     print('-------------')
#     print(loss)
