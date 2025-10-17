import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 定义 DownSample 模块
class DownSample(nn.Module):
    def __init__(self, d_model):
        super(DownSample, self).__init__()
        self.conv_down = nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')  # 转换为 (batch_size, channels, seq_len)
        down_x = self.conv_down(x)
        return rearrange(down_x, 'b d n -> b n d')  # 转回 (batch_size, seq_len, channels)


# 定义 UpSample 模块
class UpSample(nn.Module):
    def __init__(self, d_model, size):
        super(UpSample, self).__init__()
        self.conv_up = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model // 2, kernel_size=3, stride=2,
                                          padding=1)
        self.size = size

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')  # 转换为 (batch_size, channels, seq_len)
        up_x = self.conv_up(x)
        if up_x.shape[-1] != self.size:
            up_x = F.interpolate(up_x, size=self.size, mode='nearest')
        return rearrange(up_x, 'b d n -> b n d')  # 转回 (batch_size, seq_len, channels)


# 定义 RMSNorm 模块
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# 定义 PreNorm 模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 定义 FeedForward 模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 定义 MSSA 模块 (带稀疏约束的自注意力机制)
class MSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., k=10, threshold=0.1):
        super(MSSA, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 3 for Q, K, V
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * self.scale
        k = k * self.scale
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = self.attend(attn)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.to_out(out + x)


# 定义 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MSSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# 定义 Fusion 模块
class FusionModule(nn.Module):
    def __init__(self, d_model, target_dim, seq_len1, seq_len2):
        super(FusionModule, self).__init__()
        self.linear_x = nn.Linear(d_model, target_dim)  # Ensure the input dim matches d_model
        self.linear_x2 = nn.Linear(target_dim, target_dim)
        self.weight = nn.Parameter(torch.rand(1))
        self.sigmoid = nn.Sigmoid()
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2

    def forward(self, x, x2):
        if self.seq_len1 != self.seq_len2:
            x = F.interpolate(x.permute(0, 2, 1), size=self.seq_len2, mode='nearest').permute(0, 2, 1)  # Adjust seq_len only
        x_proj = self.linear_x(x)
        x2_proj = self.linear_x2(x2)
        weight = self.sigmoid(self.weight)
        return weight * x_proj + (1 - weight) * x2_proj


# 定义 PyramidTransformer
class PyramidTransformer(nn.Module):
    def __init__(self, seq_len, d_model, heads, num_layers, dim_head, mlp_dim, expand=2, dropout=0.1, bias=False):
        super(PyramidTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Ensure dimensions match across projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        seq_array = [seq_len]
        d = seq_len
        for _ in range(2):
            d = (d - 1) // 2 + 1
            seq_array.append(d)

        self.down1 = DownSample(self.d_inner)
        self.down2 = DownSample(2 * self.d_inner)
        self.down3 = DownSample(4 * self.d_inner)

        self.up1 = UpSample(8 * self.d_inner, seq_array[2])
        self.up2 = UpSample(4 * self.d_inner, seq_array[1])
        self.up3 = UpSample(2 * self.d_inner, seq_array[0])

        self.trans1 = Transformer(self.d_inner, num_layers, heads, dim_head, mlp_dim, dropout)
        self.trans2 = Transformer(2 * self.d_inner, num_layers, heads, dim_head, mlp_dim, dropout)
        self.trans3 = Transformer(4 * self.d_inner, num_layers, heads, dim_head, mlp_dim, dropout)
        self.trans4 = Transformer(8 * self.d_inner, num_layers, heads, dim_head, mlp_dim, dropout)

        self.fusion1 = FusionModule(self.d_inner, 2 * self.d_inner, seq_len, seq_array[1])
        self.fusion2 = FusionModule(2 * self.d_inner, 4 * self.d_inner, seq_array[1], seq_array[2])

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        pre_x = x
        x = self.norm(x)
        xz = self.in_proj(x)

        x, z = xz.chunk(2, dim=-1)
        x1_trans = self.trans1(x)
        x2 = self.down1(x)
        fused_x2 = self.fusion1(x, x2)
        x2_trans = self.trans2(fused_x2)
        x3 = self.down2(x2)
        fused_x3 = self.fusion2(x2, x3)
        x3_trans = self.trans3(fused_x3)
        x4 = self.down3(x3)
        x4_trans = self.trans4(x4)

        x3_sup = self.up1(x4_trans) + x3_trans
        x2_sup = self.up2(x3_sup) + x2_trans
        x1_sup = self.up3(x2_sup) + x1_trans

        z = F.silu(z)
        x_combined = x1_sup * z
        out = self.out_proj(x_combined)

        return out + pre_x


if __name__ == "__main__":
    seq_len, d_model = 128, 64
    model = PyramidTransformer(seq_len=seq_len, d_model=d_model, heads=4, num_layers=2, dim_head=32, mlp_dim=128)
    x = torch.randn(16, seq_len, d_model)
    out = model(x)
    print(out.shape)
