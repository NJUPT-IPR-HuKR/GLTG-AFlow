import torch
from torch import nn as nn

from models.modules.flow import Conv2d
import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation



def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "squeeze_invconv": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, Ft, rev: obj.invconv(z, logdet, Ft, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive",
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector
        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None
        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt
        self.sp_att = Spatialmap()
        self.channels_for_nn = in_channels // 2
        self.in_channels = in_channels


        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute

        if flow_permutation == "invconv":
            self.invconv = models.modules.Permutations.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults)
        else:
            return self.reverse_flow(input, logdet, rrdbResults)
    def get_self_attention(self, ft, z, reverse):
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

    def normal_flow(self, z, logdet, rrdbResults=None):
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)

        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)

        # 2. permute
        ftp = getConditional(rrdbResults, self.position)
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False, ftp)
        # 4. sp_att#######################SAN###########################
        z1, z2 = self.split(z)
        ftsp = getConditional(rrdbResults, self.position)
        _, z1c, _, _ = z1.shape
        conv1 = nn.Conv2d(288, z1c, kernel_size=1, bias=False).half().to(ftsp.device)  # 1x1
        ftsp = conv1(ftsp)
        ftsp = torch.cat([z1, ftsp], dim=1)
        _, c, _, _ = ftsp.shape
        # ################head####################
        layers = [Conv2d(c, 64, kernel_size=1), nn.GELU(), Conv2d(64, 64, kernel_size=1), nn.GELU()]
        head = nn.Sequential(*layers).to(ftsp.device)
        x = head(ftsp)
        # ################head####################
        attmap = self.sp_att(z, x)
        z2 = z2 * attmap  # multiply by spacial map
        logdet = logdet + self.get_logdet(attmap) * self.channels_for_nn
        z = thops.cat_feature(z1, z2)
        #######################################################################

        # 5. self#######################CSN#############
        sft = getConditional(rrdbResults, self.position)
        self_attention, slogdet = self.get_self_attention(sft, z, reverse=False)
        self_attention = self_attention.to(z.device)
        sz1, sz2 = self.split(z)
        sz1c = sz1.shape[1]
        sz2 = sz2 * self_attention
        logdet = logdet + slogdet * sz1c
        z = thops.cat_feature(sz1, sz2)
        ######################################################
        # 3. #####SCAN###############################################
        need_features = self.affine_need_features()
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft)
        ##################################################################
        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None):

        need_features = self.affine_need_features()
        img_ft = getConditional(rrdbResults, self.position)
        # 1.SCAN
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft)


        #  5.self  ##################CSN################
        sft = getConditional(rrdbResults, self.position)
        self_attention, slogdet = self.get_self_attention(sft, z, reverse=True)
        self_attention = self_attention.to(z.device)
        z1, z2 = self.split(z)
        z1c = z1.shape[1]
        z2 = z2 / self_attention
        logdet = logdet - slogdet * z1c
        z = thops.cat_feature(z1, z2)
        ################################################

        # spatt  ##################SAN################
        z1, z2 = self.split(z)
        _, z1c, _, _ = z1.shape
        conv1 = nn.Conv2d(288, z1c, kernel_size=1, bias=False).half().to(img_ft.device)  # 1x1
        ft = conv1(img_ft)
        ft = torch.cat([z1, ft], dim=1)
        _, c, _, _ = ft.shape
        # ################head####################
        layers = [Conv2d(c, 64, kernel_size=1), nn.GELU(), Conv2d(64, 64, kernel_size=1), nn.GELU()]
        head = nn.Sequential(*layers).to(ft.device)
        x = head(ft)
        # ################head####################
        attmap = self.sp_att(z, x)
        z2 = z2 / attmap
        z = thops.cat_feature(z1, z2)
        logdet = logdet - self.get_logdet(attmap) * self.channels_for_nn
        ###########################################



        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True, img_ft)
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])


class Spatialmap(nn.Module):
    def __init__(self):
        super(Spatialmap, self).__init__()
        kernel_size = 5

        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, z, ft):
        x_compress = self.compress(ft)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out + 2.) + 0.0001  # broadcasting
        return scale

    def split(self, z):
        _, c, _, _ = z.shape
        channels_for_nn = int(c/2)
        z1 = z[:, :channels_for_nn]
        z2 = z[:, channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):

        conv = self.conv.to(x.device)
        x = conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
