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


# 定义 MSSA 模块 (替换了原有的 Attention 模块)
class MSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., step_size=0.):
        super(MSSA, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()

        # 使用 FeedForward 网络进行处理
        self.FF = FeedForward(dim, inner_dim, dropout)

    def forward(self, x):
        # 计算 QKV
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        # 计算点积注意力
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 使用注意力权重进行加权
        out = torch.matmul(attn, w)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x  # 残差连接

        # 使用 FeedForward 进行进一步处理
        out = self.FF(out)
        return out


# 定义 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])  # Transformer 层列表
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MSSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 替换 Attention 为 MSSA
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x  # 残差连接
        return x


# 定义 PyramidTransformer 模块
class PyramidTransformer(nn.Module):
    def __init__(self, seq_len, d_model, heads, num_layers, dim_head, mlp_dim, expand=2, dropout=0.1, bias=False):
        super(PyramidTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

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

        self.norm = RMSNorm(d_model)

    def forward(self, x):
        pre_x = x
        x = self.norm(x)
        xz = self.in_proj(x)

        x, z = xz.chunk(2, dim=-1)

        x1_trans = self.trans1(x)
        x2 = self.down1(x1_trans)
        x2_trans = self.trans2(x2)
        x3 = self.down2(x2)
        x3_trans = self.trans3(x3)
        x4 = self.down3(x3)
        x4_trans = self.trans4(x4)

        x3_sup = self.up1(x4_trans) + x3_trans
        x2_sup = self.up2(x3_sup) + x2_trans
        x1_sup = self.up3(x2_sup) + x1_trans

        z = F.silu(z)
        x_combined = x1_sup * z
        out = self.out_proj(x_combined)

        return out + pre_x


# 主程序
if __name__ == "__main__":
    device = torch.device("cpu")  # 强制设置为 CPU

    seq_len, d_model = 128, 64
    model = PyramidTransformer(seq_len=seq_len, d_model=d_model, heads=4, num_layers=2, dim_head=32, mlp_dim=128,
                               dropout=0.1)
    model.to(device)  # 将模型移到 CPU

    x = torch.randn(16, seq_len, d_model).to(device)  # 将输入移到 CPU
    out = model(x)
    print(out.shape)  # 输出形状 (16, seq_len, d_model)
