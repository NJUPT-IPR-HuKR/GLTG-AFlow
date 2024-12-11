import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################

class FeedForward_new(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_new, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.c31 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.c32 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.c33 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fc = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features // 16, hidden_features, 1, bias=False)
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        avg = torch.mean(x1, dim=1, keepdim=True)

        max_v, max_index = torch.max(x1, dim=1, keepdim=True)
        x3 = torch.cat([avg, max_v], dim=1)

        x3 = self.c33(self.c32(self.c31(x3)))
        x = F.sigmoid(x3) * F.gelu(x2)
        x = self.project_out(x)

        return x

class MaskMap(nn.Module):
    def __init__(self, num_heads, N=4, LayerNorm_type='WithBias', bias=False, ):
        super(MaskMap, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.num_heads = num_heads
        self.N = N

    def forward(self, mask):
        mask = mask.float()
        q = rearrange(mask, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        k = rearrange(mask, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        attn_mask = (q @ k.transpose(-2, -1)) * self.temperature
        attn_mask = torch.nn.functional.normalize(attn_mask, dim=-1)

        return attn_mask
##########################################################################

##########################################################################
class Mask(nn.Module):
    def __init__(self, dim, num_heads, N=4, LayerNorm_type='WithBias', bias=False, ):
        super(Mask, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.num_heads = num_heads
        self.N = N
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_conv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        mask_in = mask
        n11, c11, h11, w11 = x.shape
        h_pad = 4 - h11 % 4 if not h11 % 4 == 0 else 0
        w_pad = 4 - w11 % 4 if not w11 % 4 == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')


        _, _, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        _, _, v = qkv.chunk(3, dim=1)

        v = rearrange(v, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        q = rearrange(mask_in, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)  # N^2,HW/N^2
        k = rearrange(mask_in, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h1 w1) (h w) -> b (head c) (h1 h) (w1 w)', h=int(h / self.N), w=int(w / self.N),
                        h1=self.N, w1=self.N, head=self.num_heads)
        out = self.project_out(out)
        out = out[:, :, :h11, :w11]
        return out



class Attention_inter(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_inter, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_intra(nn.Module):
    def __init__(self, dim, num_heads, N=4, LayerNorm_type='WithBias', bias=False, ):
        super(Attention_intra, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.num_heads = num_heads
        self.N = N
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.get_attn_m = MaskMap(num_heads,N=self.N)

    def forward(self, x, mask):
        n11, c11, h11, w11 = x.shape
        h_pad = 4 - h11 % 4 if not h11 % 4 == 0 else 0
        w_pad = 4 - w11 % 4 if not w11 % 4 == 0 else 0
        x_1 = F.pad(x, (0, w_pad, 0, h_pad), 'reflect')
        _, _, h, w = x_1.shape
        qkv = self.qkv_dwconv(self.qkv(x_1))
        q, k, v = qkv.chunk(3, dim=1)

        v = rearrange(v, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        q = rearrange(q, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)  # N^2,HW/N^2
        k = rearrange(k, 'b (head c) (h1 h) (w1 w) -> b head c (h1 w1) (h w)', h1=self.N, w1=self.N,
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn_x = (q @ k.transpose(-2, -1)) * self.temperature
        attn_x = attn_x.softmax(dim=-1)
        attn_m = self.get_attn_m(mask)
        attn_m = attn_m.softmax(dim=-1)
        attn = (attn_x + attn_m).softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h1 w1) (h w) -> b (head c) (h1 h) (w1 w)', h=int(h / self.N), w=int(w / self.N),
                        h1=self.N, w1=self.N, head=self.num_heads)
        out = self.project_out(out)
        out = out[:, :, :h11, :w11]
        return out


##########################################################################
class TransformerBlockWithMask(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlockWithMask, self).__init__()

        self.mask = Mask(dim, num_heads)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_inter(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn_new = FeedForward_new(dim, ffn_expansion_factor, bias)
        self.attn_intra = Attention_intra(dim, num_heads, N=4)
        self.conv_cat1 = nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=True)
        self.conv_cat2 = nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=True)

    def forward(self, x, mask):
        ################### intra & inter  #########################
        x_n = self.norm1(x)
        fea1 = x + self.attn(x_n)
        fea2 = fea1 + self.attn_intra(self.norm2(fea1), mask)
        out = fea2 + self.ffn_new(self.norm3(fea2))
        return out
        #############################################################

############################################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_inter(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##############################################################################################
class TransformerBlock_1(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers=6, dim=3, num_heads=4, ffn_expansion_factor=2.66, bias=True,
                 LayerNorm_type='WithBias'):
        # 2048
        super().__init__()

        self.layer_stack = nn.ModuleList([
            TransformerBlockWithMask(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(n_layers)])

    def forward(self, x, mask):
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
class Restormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 4, 6, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):

        super(Restormer, self).__init__()
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = TransformerBlock_1(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                                 bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock_1(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock_1(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[2])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4


        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))


        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)


        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.Conv2d(64, 128, 3, 2, 1),
                                                 nn.Conv2d(128, 192, 3, 2, 1), nn.Sigmoid())
        self.reduce_chan0 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan1 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan2 = nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 2), kernel_size=1, bias=bias)

    def downmask(self, mask, k):
        _, _, h, w = mask.shape
        m = torch.squeeze(mask, 1)
        b, h, w = m.shape
        m = torch.chunk(mask, b, dim=0)[0]
        m = torch.squeeze(m)
        size = (int(h / k), int(w / k))
        mc = m.cpu()
        m_n = mc.numpy()
        m = cv2.resize(m_n, size, interpolation=cv2.INTER_LINEAR)
        m[m <= 0.2] = 0
        m[m > 0.2] = 1
        m = torch.from_numpy(m)
        m = torch.unsqueeze(m, 0)
        _, h, w = m.shape
        h_pad = 4 - h % 4 if not h % 4 == 0 else 0
        w_pad = 4 - w % 4 if not w % 4 == 0 else 0
        m = F.pad(m, (0, w_pad, 0, h_pad), 'reflect')
        m = torch.unsqueeze(m, 0)
        m = m.expand(b, self.dim * k, -1, -1)
        out_mask = m.cuda()
        return out_mask

    def forward(self, inp_img, mask):
        mc = mask.shape[1]

        m = torch.chunk(mask, mc, dim=1)[0]
        b, _, _, _ = inp_img.shape
        result = {}


        m_1 = m.expand(b, self.dim, -1, -1)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1, m_1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        m_2 = self.downmask(m, int(2))
        out_enc_level2 = self.encoder_level2(inp_enc_level2, m_2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        m_3 = self.downmask(m, int(2 ** 2))
        out_enc_level3 = self.encoder_level3(inp_enc_level3, m_3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent = self.latent(inp_enc_level4)
        result['fea_up0'] = self.reduce_chan0(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        result['fea_up1'] = self.reduce_chan1(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        result['fea_up2'] = self.reduce_chan2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        result['cat_f'] = out_dec_level1

        out_dec_level0 = self.output(out_dec_level1) + inp_img
        result['color_map'] = self.fine_tune_color_map(out_dec_level0)
        return result

