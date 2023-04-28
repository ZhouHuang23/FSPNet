import torch
import torch.nn as nn
import torch.nn.functional as F
import vit


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False): # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x): # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class Converter(nn.Module):
    def __init__(self, dim_in, dim_temp=384, img_size=384, mids=4):
        super(Converter, self).__init__()

        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_temp = dim_temp

        self.num_n = mids * mids

        self.conv_fc = nn.Conv2d(self.dim_in * 2, self.dim_temp, kernel_size=1)

        # f1
        self.norm_layer_f1 = nn.LayerNorm(dim_in)
        self.conv_f1_Q = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.conv_f1_K = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.ap_f1 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f1 = GCN(num_state=self.dim_temp, num_node=self.num_n)
        self.conv_f1_extend = nn.Conv2d(self.dim_temp, self.dim_in, kernel_size=1, bias=False)

        # f2
        self.norm_layer_f2 = nn.LayerNorm(dim_in)
        self.conv_f2_Q = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.conv_f2_K = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.ap_f2 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f2 = GCN(num_state=self.dim_temp, num_node=self.num_n)
        self.conv_f2_extend = nn.Conv2d(self.dim_temp, self.dim_in, kernel_size=1, bias=False)

    def forward(self, token_pair):
        # tokens list 12x[8,578,768]
        bs, num_token, chs = token_pair[0].shape
        tokens_ls = []
        for index in range(len(token_pair) // 2):
            f1_ = self.norm_layer_f1(token_pair[index * 2][:, 2:, :])  # [8,576,768]
            f2_ = self.norm_layer_f2(token_pair[index * 2 + 1][:, 2:, :])  # [8,576,768]
            f1_ = f1_.permute(0, 2, 1).view(bs, chs, int(self.img_size // 16), int(self.img_size // 16)).contiguous()
            # [8,768,24,24]
            f2_ = f2_.permute(0, 2, 1).view(bs, chs, int(self.img_size // 16), int(self.img_size // 16)).contiguous()
            # [8,768,24,24]
            f1, f2 = f1_, f2_  # [8,768,24,24] / [8,768,24,24]
            fc = self.conv_fc(torch.cat((f1, f2), dim=1))  # [8,384,24,24]
            fc_att = torch.nn.functional.softmax(fc, dim=1)[:, 1, :, :].unsqueeze(1)  # [8,1,24,24]

            # f1 pass
            f1_Q = self.conv_f1_Q(f1).view(bs, self.dim_temp, -1).contiguous()  # [8,384,576] [bs,chs,24*24]
            f1_K = self.conv_f1_K(f1)  # [8,384,24,24]
            f1_masked = f1_K * fc_att  # [8,384,24,24]
            f1_V = self.ap_f1(f1_masked)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)  # [8,384,16]

            f1_proj_reshaped = torch.matmul(f1_V.permute(0, 2, 1), f1_K.reshape(bs, self.dim_temp, -1))  # [8,16,576]
            f1_proj_reshaped = torch.nn.functional.softmax(f1_proj_reshaped, dim=1)  # [8,16,576] Tv

            f1_rproj_reshaped = f1_proj_reshaped  # [8,16,576]
            f1_n_state = torch.matmul(f1_Q, f1_proj_reshaped.permute(0, 2, 1))  # [16,384,16] Ta

            f1_n_rel = self.gcn_f1(f1_n_state)  # [16,384,16]
            f1_state_reshaped = torch.matmul(f1_n_rel, f1_rproj_reshaped)  # [16,384,576]
            f1_state = f1_state_reshaped.view(bs, self.dim_temp, *f1.size()[2:])  # [16,384,24,24]
            f1_out = f1_ + (self.conv_f1_extend(f1_state))  # [16,768,24,24]

            # f2 pass
            f2_Q = self.conv_f2_Q(f2).view(bs, self.dim_temp, -1).contiguous()  # [8,384,576] [bs,chs,24*24]
            f2_K = self.conv_f2_K(f2)  # [8,384,24,24]
            f2_masked = f2_K * fc_att  # [8,384,24,24]
            f2_V = self.ap_f2(f2_masked)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)  # [8,384,16]

            f2_proj_reshaped = torch.matmul(f2_V.permute(0, 2, 1), f2_K.reshape(bs, self.dim_temp, -1))  # [8,16,576]
            f2_proj_reshaped = torch.nn.functional.softmax(f2_proj_reshaped, dim=1)  # [8,16,576]

            f2_rproj_reshaped = f2_proj_reshaped  # [8,16,576]
            f2_n_state = torch.matmul(f2_Q, f2_proj_reshaped.permute(0, 2, 1))  # [16,384,16]

            f2_n_rel = self.gcn_f2(f2_n_state)  # [16,384,16]
            f2_state_reshaped = torch.matmul(f2_n_rel, f2_rproj_reshaped)  # [16,384,576]
            f2_state = f2_state_reshaped.view(bs, self.dim_temp, *f2.size()[2:])  # [16,384,24,24]
            f2_out = f2_ + (self.conv_f2_extend(f2_state))  # [16,768,24,24]

            tokens_ls.extend([f1_out, f2_out])

        return tokens_ls


class UpSampling2x(nn.Module): 
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, features):
        return self.up_module(features)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, start=False):  # 768, 384
        super(GroupFusion, self).__init__()
        temp_chs = in_chs
        if start:
            in_chs = in_chs
        else:
            in_chs *= 2

        self.gf1 = nn.Sequential(nn.Conv2d(in_chs, temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))

        self.gf2 = nn.Sequential(nn.Conv2d((temp_chs + temp_chs), temp_chs, 1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(temp_chs, temp_chs, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(temp_chs),
                                 nn.ReLU(inplace=True))
        self.up2x = UpSampling2x(temp_chs, out_chs)

    def forward(self, f_r, f_l):
        f_r = self.gf1(f_r)  # chs 768
        f12 = self.gf2(torch.cat((f_r, f_l), dim=1))  # chs 768
        return f12, self.up2x(f12)


class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)


class Model(nn.Module):
    def __init__(self, ckpt, img_size=384):
        super(Model, self).__init__()
        self.encoder = vit.deit_base_distilled_patch16_384()
        if ckpt is not None:
            ckpt = torch.load(ckpt, map_location='cpu')
            msg = self.encoder.load_state_dict(ckpt["model"], strict=False)
            print("====================================")
            print(msg)

        self.img_size = img_size
        self.vit_chs = 768

        self.group_converter_0 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_1 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_2 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_3 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_4 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_5 = Converter(dim_in=self.vit_chs, img_size=self.img_size)

        self.gf1_1 = GroupFusion(768, 384)
        self.gf1_2 = GroupFusion(768, 384)
        self.gf1_3 = GroupFusion(768, 384)
        self.gf1_4 = GroupFusion(768, 384)
        self.gf1_5 = GroupFusion(768, 384)
        self.gf1_6 = GroupFusion(768, 384, start=True)

        self.gf2_1 = GroupFusion(384, 192)
        self.gf2_2 = GroupFusion(384, 192)
        self.gf2_3 = GroupFusion(384, 192, start=True)

        self.gf3_1 = GroupFusion(192, 192)
        self.gf3_2 = GroupFusion(192, 192, start=True)

        self.gf4_1 = GroupFusion(192, 192, start=True)

        self.out1 = OutPut(in_chs=768, scale=16)
        self.out2 = OutPut(in_chs=384, scale=8)
        self.out3 = OutPut(in_chs=192, scale=4)
        self.out4 = OutPut(in_chs=192)

    def group_converter_fn(self, tokens):
        group_converter_ls = [self.group_converter_0, self.group_converter_1, self.group_converter_2,
                              self.group_converter_3, self.group_converter_4, self.group_converter_5]
        tokens_ls = []
        for index in range(len(tokens) // 2):
            token_pair = [tokens[index * 2], tokens[index * 2 + 1]]
            token_pair_out = group_converter_ls[index](token_pair)
            tokens_ls.extend(token_pair_out)

        return tokens_ls

    def group_pyramid_decode(self, feature):
        # list 12x[8,768,24,24]
        # layer1 out
        f1_6_l, f2_6 = self.gf1_6(feature[-1], feature[-2])
        f1_5_l, f2_5 = self.gf1_5(torch.cat((feature[-3], f1_6_l), dim=1), feature[-4])
        f1_4_l, f2_4 = self.gf1_4(torch.cat((feature[-5], f1_5_l), dim=1), feature[-6])
        f1_3_l, f2_3 = self.gf1_3(torch.cat((feature[-7], f1_4_l), dim=1), feature[-8])
        f1_2_l, f2_2 = self.gf1_2(torch.cat((feature[-9], f1_3_l), dim=1), feature[-10])
        f1_1_o, f2_1 = self.gf1_1(torch.cat((feature[-11], f1_2_l), dim=1), feature[-12])  # f1_1_l [bs,768,24,24]
        # layer2 out
        f2_3_l, f3_3 = self.gf2_3(f2_6, f2_5)
        f2_2_l, f3_2 = self.gf2_2(torch.cat((f2_4, f2_3_l), dim=1), f2_3)
        f2_1_o, f3_1 = self.gf2_1(torch.cat((f2_2, f2_2_l), dim=1), f2_1)  # f2_1_l [bs,384,48,48]
        # layer3 out
        f3_2_l, f4_2 = self.gf3_2(f3_3, f3_2)
        f3_1_o, f4_1 = self.gf3_1(torch.cat((f3_2, f3_2_l), dim=1), f3_1)  # f3_1_l [bs,192,96,96]
        # layer4 out
        _, f5_1 = self.gf4_1(f4_2, f4_1)
        return f1_1_o, f2_1_o, f3_1_o, f5_1

    def pred_out(self, gpd_outs):
        return self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2]), self.out4(gpd_outs[3])

    def forward(self, img):
        # B Seq
        B, C, H, W = img.size()
        x = self.encoder(img)  # list 12x[8,576,768]
        feature = self.group_converter_fn(x)
        gpd_outs = self.group_pyramid_decode(feature)
        return self.pred_out(gpd_outs)
