
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from models.modules.flow import Conv2d
from models.modules import thops


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False, Ft=None):
        super().__init__()
        self.exteact_f = None
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.channels_for_nn = num_channels // 2
        self.ft = Ft


    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2
    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        pixels = thops.pixels(input)
        dlogdet = torch.tensor(float('inf'))
        while torch.isinf(dlogdet):
            try:
                dlogdet = torch.slogdet(self.weight)[1] * pixels
            except Exception as e:
                print(e)
                dlogdet = \
                    torch.slogdet(
                        self.weight + (self.weight.mean() * torch.randn(*self.w_shape).to(input.device) * 0.001))[
                        1] * pixels
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            try:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            except:
                weight = torch.inverse(self.weight.double()+ (self.weight.mean() * torch.randn(*self.w_shape).to(input.device) * 0.001).float() \
                    .view(w_shape[0], w_shape[1], 1, 1))
        return weight, dlogdet

    def get_self_attention(self, ft, reverse, z):
        z1, z2 = self.split(z)
        _, z1c, _, _ = z1.shape
        ft = ft.float()
        conv1 = nn.Conv2d(288, z1c, kernel_size=1, bias=False).half().to(ft.device)  # 1x1
        ft = conv1(ft)
        ft = torch.cat([z1, ft], dim=1)
        ft_b, ft_c, ft_w, ft_h = ft.shape
        # ################head####################
        layers1 = [Conv2d(ft_c, 64, kernel_size=1), nn.GELU(), Conv2d(64, 64, kernel_size=1), nn.GELU()]
        head1 = nn.Sequential(*layers1).to(ft.device)
        ft = head1(ft)
        # ################head####################
        b, c, fh, fw = ft.shape  # feature height and feature width
        fc1 = nn.Sequential(
            nn.Conv2d(c, c // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 8, c, 1, bias=False)
        ).to(ft.device)
        fc2 = nn.Sequential(
            nn.Conv2d(c, c // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 8, c, 1, bias=False)
        ).to(ft.device)
        wa = (fc1(torch.mean(ft, dim=3, keepdim=True))).squeeze(dim=3).transpose(-2, -1)  # b,w,c
        ha = (fc2(torch.mean(ft, dim=2, keepdim=True))).squeeze(dim=2)  # b,c,h
        self_attention = wa @ ha

        self_attention = (torch.sigmoid(self_attention+2.) + 0.0001).unsqueeze(dim=1)
        slogdet = thops.sum(torch.log(self_attention), dim=[1, 2, 3])
        return self_attention, slogdet

    def forward(self, z, logdet=None, reverse=False, ft=None):
        """
        log-det = log|abs(|W|)| * pixels
        """
        self_attention, slogdet = self.get_self_attention(ft, reverse, z)  # SELF_ATTENTION b,c,c,1,1 slogdet (b,)
        self_attention = self_attention.to(z.device)
        weight, dlogdet = self.get_weight(z, reverse)

        if not reverse:  # 1x1 invert conv

            z = F.conv2d(z, weight)
            if logdet is not None:
                logdet = logdet + dlogdet

            return z, logdet
        else:

            z = F.conv2d(z, weight)
            if logdet is not None:
                logdet = logdet - dlogdet

            return z, logdet


