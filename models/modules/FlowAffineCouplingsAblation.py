
import torch
from torch import nn as nn
import torch.nn.functional as F

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 288)
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn
        self.sp_att = Spatialmap()


        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = NN_F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = NN_F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = NN_F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Curve conditional
            if self.le_curve:
                alpha = self.fCurve(ft)
                alpha = torch.relu(alpha) + self.affine_eps
                logdet = logdet + thops.sum(torch.log(alpha * torch.pow(z.abs(), alpha - 1)) + self.affine_eps)
                z = torch.pow(z.abs(), alpha) * z.sign()

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)

            output = z


        else:
            z = input
            z = z.to(ft.device)
            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Curve conditional
            if self.le_curve:

                alpha = self.fCurve(ft)
                alpha = torch.relu(alpha) + self.affine_eps
                z = torch.pow(z.abs(), 1 / alpha) * z.sign()

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z1 = z1.to(ft.device)
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift



    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))

        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)

        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class Spatialmap(nn.Module):
    def __init__(self):
        super(Spatialmap, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out + 2.) + 0.0001  # broadcasting
        return scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out + x

class NN_F(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, image_shape=None, H=None, W=None):
        super(NN_F, self).__init__()
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=kernel_hidden), nn.GELU()]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_hidden))
            layers.append(nn.GELU())
        layers.append(SCAN(gate_channels=hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=kernel_hidden))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ChannelGate_conv_wh(nn.Module):
    def __init__(self, gate_channels, pool_types=('avg', 'max')):
        super(ChannelGate_conv_wh, self).__init__()
        self.gate_channels = gate_channels
        self.conv = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels * 2, kernel_size=3, stride=1, padding=1,
                                    groups=gate_channels * 2, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1, bias=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        global scale_w
        for pool_type in self.pool_types:
            if pool_type == 'w_avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                channel_att_raw = self.conv(avg_pool)
                scale_w = torch.sigmoid(channel_att_raw).expand_as(x)
            elif pool_type == 'h_max':
                avg_pool11 = F.max_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                channel_att_raw1 = self.conv(avg_pool11)
                scale_h = torch.sigmoid(channel_att_raw1).expand_as(x)
            elif pool_type == 'w_max':
                avg_pool = F.max_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                channel_att_raw = self.conv(avg_pool)
                scale_w = torch.sigmoid(channel_att_raw).expand_as(x)
            elif pool_type == 'h_avg':
                avg_pool11 = F.avg_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                channel_att_raw1 = self.conv(avg_pool11)
                scale_h = torch.sigmoid(channel_att_raw1).expand_as(x)
            elif pool_type == 'c_avg':
                avg_cp = torch.mean(x, dim=0, keepdim=False)
                avg_cp = self.conv(avg_cp)
                scale_w = torch.sigmoid(avg_cp).expand_as(x)
            elif pool_type == 'c_max':
                max_cp, _ = torch.max(x, dim=0, keepdim=False)
                max_cp = self.conv(max_cp)
                scale_h = torch.sigmoid(max_cp).expand_as(x)
        scale = scale_w + scale_h

        return x * scale


class SCAN(nn.Module):
    def __init__(self, gate_channels):
        super(SCAN, self).__init__()
        self.ChannelGate_wh0 = ChannelGate_conv_wh(gate_channels, pool_types=('c_avg', 'c_max'))
        self.ChannelGate_wh = ChannelGate_conv_wh(gate_channels, pool_types=('w_avg', 'h_max'))
        self.ChannelGate_wh1 = ChannelGate_conv_wh(gate_channels, pool_types=('w_max', 'h_avg'))

    def forward(self, x):
        x_out_1 = self.ChannelGate_wh0(x)
        x_out_2 = self.ChannelGate_wh(x_out_1)
        x_out_3 = self.ChannelGate_wh1(x_out_2)

        return x_out_3 + x
