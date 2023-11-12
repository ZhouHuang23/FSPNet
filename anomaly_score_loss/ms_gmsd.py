import torch
from torch import nn
from torch.nn import functional as F


class GMSD_Loss(nn.Module):
    # Refer to http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

    def __init__(self, channels=3):
        super(GMSD_Loss, self).__init__()
        self.channels = channels
        dx = (torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1,
                                                                                                        1)
        dy = (torch.Tensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / 3.).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1,
                                                                                                        1)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.dy = nn.Parameter(dy, requires_grad=False)
        self.aveKernel = nn.Parameter(torch.ones(channels, 1, 2, 2) / 4., requires_grad=False)

    def gmsd(self, img1, img2, T=170):
        # Y1 = F.conv2d(img1, self.aveKernel, stride=2, padding=0, groups=self.channels)
        # Y2 = F.conv2d(img2, self.aveKernel, stride=2, padding=0, groups=self.channels)
        Y1 = F.conv2d(img1, self.aveKernel, stride=1, padding=0, groups=self.channels)
        Y2 = F.conv2d(img2, self.aveKernel, stride=1, padding=0, groups=self.channels)

        IxY1 = F.conv2d(Y1, self.dx, stride=1, padding=1, groups=self.channels)
        IyY1 = F.conv2d(Y1, self.dy, stride=1, padding=1, groups=self.channels)
        gradientMap1 = torch.sqrt(IxY1 ** 2 + IyY1 ** 2 + 1e-12)

        IxY2 = F.conv2d(Y2, self.dx, stride=1, padding=1, groups=self.channels)
        IyY2 = F.conv2d(Y2, self.dy, stride=1, padding=1, groups=self.channels)
        gradientMap2 = torch.sqrt(IxY2 ** 2 + IyY2 ** 2 + 1e-12)

        quality_map = (2 * gradientMap1 * gradientMap2 + T) / (gradientMap1 ** 2 + gradientMap2 ** 2 + T)
        score = torch.std(quality_map.view(quality_map.shape[0], -1), dim=1)
        return score, quality_map

    def forward(self, y, x, as_loss=True):
        assert x.shape == y.shape
        x = x * 255
        y = y * 255
        if as_loss:
            score, map = self.gmsd(x, y)
            return score.mean(), map
        else:
            with torch.no_grad():
                score, map = self.gmsd(x, y)
            return score, map


class MSGMSD_Loss(torch.nn.Module):
    def __init__(self, num_scales=3):
        super(MSGMSD_Loss, self).__init__()
        use_cuda = torch.cuda.is_available()
        device=torch.device('cuda' if use_cuda else 'cpu')
        self.num_scales = num_scales
        self.model = GMSD_Loss().to(device)

    def forward(self, img_1, img_2, as_loss=True):
        b, c, h, w = img_1.shape
        msgmsd_map = 0
        msgmsd_loss = 0

        for scale in range(self.num_scales):
            if scale > 0:
                img_1 = F.avg_pool2d(img_1, kernel_size=2, stride=2, padding=0)
                img_2 = F.avg_pool2d(img_2, kernel_size=2, stride=2, padding=0)

            score, map = self.model(img_1, img_2, as_loss=as_loss)
            msgmsd_map += F.interpolate(map, size=(h, w), mode="bilinear", align_corners=False)
            msgmsd_loss += score

        if as_loss:
            return msgmsd_loss / self.num_scales
        else:
            return torch.mean(1 - msgmsd_map / self.num_scales, axis=1).unsqueeze(1)


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

#     model = MSGMSD(device=device)
#     loss = model(img_1, img_2, as_loss=True)
#     print(loss)
#     print('--------------------')
#     map = model(img_1, img_2, as_loss=False)
#     print(map)
#     print(map.shape)
#     print(torch.min(map))
#     print(torch.max(map))
#     save_image(unorm_image(img_1, opt.data_mean, opt.data_std), '1.png', nrow=3)
#     save_image(unorm_image(img_2, opt.data_mean, opt.data_std), '2.png', nrow=3)
#     save_image(map, '3.png', nrow=3)