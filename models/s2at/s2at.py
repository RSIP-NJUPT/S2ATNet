import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from .cfsm import CFSM


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5  # 1/sqrt(dim)

        # Wq,Wk,Wv for each vector, thats why *3
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads  # x.shape=[64,5,64]
        # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q=k=v.shape=[64,5,64]->[64,8,5,8]
        # split into multi head attentions
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # [64,8,5,5]
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float("-inf"))
            del mask

        # follow the softmax,q,d,v equation in the paper
        attn = dots.softmax(dim=-1)

        # product of v times whatever inside softmax, [64,8,5,8]
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        # concat heads into one matrix, ready for next encoder block, [ 64,5,64]
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.nn1(out)
        out = self.do1(out)
        return out  # [ 64,5,64]


class SpatialTransformer(nn.Module):
    def __init__(self, dim, num_tokens, depth=1, heads=8, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            LayerNorm(dim, Attention(dim, heads=heads, dropout=dropout))
                        ),
                        Residual(LayerNorm(dim, MLP(dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention, [64,5,64]
            x = mlp(x)  # go to MLP, [64,5,64]
        return x


class GSAttention(nn.Module):
    """global spectral attention (GSA)

    Args:
        num_tokens (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, num_tokens, heads=8, dropout=0.1):
        super().__init__()
        # self.heads = heads
        self.scale = num_tokens**-0.5  # 1/sqrt(num_tokens)

        # Wq,Wk,Wv for each vector, thats why *3
        self.to_qkv = nn.Linear(num_tokens, num_tokens * 3, bias=True)
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(num_tokens, num_tokens)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, d = x.shape  # x.shape=[64,5,64]
        x = x.permute(0, 2, 1)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # 64,64,5*3

        dots = q @ k.transpose(1, 2) * self.scale  # [64,5*3,5*3]
        attn = dots.softmax(dim=-1)

        out = attn @ v  # 64,64,5
        out = self.nn1(out)
        out = self.do1(out).permute(0, 2, 1)
        return out  # [ 64,5,64]


class SpectralTransformer(nn.Module):
    def __init__(self, dim, num_tokens, depth=1, heads=8, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            LayerNorm(
                                dim,
                                GSAttention(
                                    num_tokens=num_tokens, heads=heads, dropout=dropout
                                ),
                            )
                        ),
                        Residual(LayerNorm(dim, MLP(dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention, [64,5,64]
            x = mlp(x)  # go to MLP, [64,5,64]
        return x


class DualBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, drop_path=0.0):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)
        self.q_proxy_ln = nn.LayerNorm(dim)

        self.p_ln = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path * 1.0) if drop_path > 0.0 else nn.Identity()

        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim),
        )
        self.proxy_ln = nn.LayerNorm(dim)

        self.qkv_proxy = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3))

        layer_scale_init_value = 1e-6
        self.gamma1 = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.gamma2 = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.gamma3 = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def selfatt(self, hsi):
        B, N, C = hsi.shape
        qkv = (
            self.qkv_proxy(hsi)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        hsi = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return hsi

    def forward(self, hsi, lidar):
        # hsi=hsi, lidar=Lidar
        hsi = hsi + self.drop_path(self.gamma1 * self.selfatt(hsi))

        B, N, C = lidar.shape
        B_p, N_p, C_p = hsi.shape
        q = (
            self.q(lidar)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q_hsi = (
            self.q_proxy(self.q_proxy_ln(hsi))
            .reshape(B_p, N_p, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        kv_hsi = (
            self.kv_proxy(lidar)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        kp, vp = kv_hsi[0], kv_hsi[1]
        attn = (q_hsi @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _hsi = (attn @ vp).transpose(1, 2).reshape(B, N_p, C) * self.gamma2
        hsi = hsi + self.drop_path(_hsi)
        hsi = hsi + self.drop_path(self.gamma3 * self.mlp_proxy(self.p_ln(hsi)))

        kv = (
            self.kv(self.proxy_ln(hsi))
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        lidar = (attn @ v).transpose(1, 2).reshape(B, N, C)
        lidar = self.proj(lidar)
        return hsi, lidar  # hsi,lidar


class SuperXuan(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        num_tokens,
        heads,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SpatialTransformer(
                            dim=dim,
                            num_tokens=num_tokens,
                            depth=depth,
                            heads=heads,
                            dropout=dropout,
                        ),
                        SpectralTransformer(
                            dim=dim,
                            num_tokens=num_tokens,
                            depth=depth,
                            heads=heads,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            dim=dim,
                            num_tokens=num_tokens,
                            depth=depth,
                            heads=heads,
                            dropout=dropout,
                        ),
                        DualBlock(dim=dim, num_heads=heads),
                    ]
                )
            )

    def forward(self, h_tokens, l_tokens):
        for h_enc1, h_enc2, l_enc, dual_block in self.layers:
            h_tokens, l_tokens = h_enc2(h_enc1(h_tokens)), l_enc(
                l_tokens
            )  # , [64,5,64]
            h_tokens, l_tokens = dual_block(h_tokens, l_tokens)  # , [64,5,64]

        return h_tokens, l_tokens


class S2ATNet(nn.Module):
    def __init__(
        self,
        num_classes,
        hsi_in_channels=30,
        lidar_in_channels=1,
        num_tokens=81,
        dim=64,
        heads=8,
        depth=1,
        dropout=0.1,
    ):
        super(S2ATNet, self).__init__()
        self.dim = dim
        self.bs = 64
        # self.hsi_conv3d = nn.Conv3d(1, out_channels, kernel_size, padding=kernel_size // 2)
        self.hsi_conv = nn.Conv2d(hsi_in_channels, dim, 1)
        self.lsk_hsi = CFSM(dim)

        self.lidar_conv = nn.Conv2d(lidar_in_channels, dim, 1)
        self.lsk_lidar = CFSM(dim)

        self.superxuan = SuperXuan(
            depth=depth,
            dim=dim,
            num_tokens=num_tokens,
            heads=heads,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(4 * dim), nn.Linear(4 * dim, num_classes)
        )

    def forward(self, x1, x2, mask=None):

        # x1 = self.hsi_conv3d(x1)  # c=64
        # x1 = rearrange(x1, "b c d h w -> b (c d) h w")
        x1 = self.hsi_conv(x1.squeeze(1))  # x1.shape=[64,30,11,11]->[64,64,11,11]
        x1 = self.lsk_hsi(x1)  # x1.shape=[64,64,11,11]->[64,64,11,11]
        # # x1.shape=[64,64,11,11]->[64,121,64]
        x1 = rearrange(x1, "b c h w -> b (h w) c")

        # x2.shape=[64,1,11,11]->[64,64,11,11]
        x2 = self.lidar_conv(x2)
        # x2.shape=[64,64,11,11]->[64,64,11,11]
        x2 = self.lsk_lidar(x2)
        # x2.shape=[64,64,11,11]->[64,121,64]
        x2 = rearrange(x2, "b c h w -> b (h w) c")

        x1, x2 = self.superxuan(x1, x2)
        x = torch.cat([x1, x2], dim=-1)
        x_ = x.transpose(-1, -2)  # b c n
        y1 = F.adaptive_avg_pool1d(x_, output_size=1).squeeze(-1)
        y2 = F.adaptive_max_pool1d(x_, output_size=1).squeeze(-1)
        x = torch.cat((y1, y2), dim=-1)

        x = self.mlp_head(x)

        return x  # b c


if __name__ == "__main__":
    model = S2ATNet(num_classes=8)
    input1 = torch.randn(64, 1, 30, 11, 11)
    input2 = torch.randn(64, 1, 11, 11)
    x = model(input1, input2)
    print(x.size())
