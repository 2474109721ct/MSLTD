import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# 前馈网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Transformer 层
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv_down = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b l d -> b d l')  # 转换为 (batch, channels, seq_len)
        down_x = self.conv_down(x)
        return rearrange(down_x, 'b d l -> b l d')  # 转回 (batch, seq_len, channels)


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv_up = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, target_size):
        x = rearrange(x, 'b l d -> b d l')  # 转换为 (batch, channels, seq_len)
        up_x = self.conv_up(x)

        # 确保上采样后的序列长度与目标一致
        if up_x.shape[-1] != target_size:
            up_x = F.interpolate(up_x, size=target_size, mode='nearest')

        return rearrange(up_x, 'b d l -> b l d')  # 转回 (batch, seq_len, channels)


# 金字塔 Transformer 模块
class PyramidTransformer(nn.Module):
    def __init__(self, seq_len, d_model, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(PyramidTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Transformer Blocks
        self.transformer1 = TransformerBlock(dim=d_model, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.transformer2 = TransformerBlock(dim=2 * d_model, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.transformer3 = TransformerBlock(dim=4 * d_model, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.transformer4 = TransformerBlock(dim=8 * d_model, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)

        # Downsampling and Upsampling layers
        self.down_1 = DownSample(in_channels=d_model, out_channels=2 * d_model)
        self.down_2 = DownSample(in_channels=2 * d_model, out_channels=4 * d_model)
        self.down_3 = DownSample(in_channels=4 * d_model, out_channels=8 * d_model)

        self.up1 = UpSample(in_channels=8 * d_model, out_channels=4 * d_model)
        self.up2 = UpSample(in_channels=4 * d_model, out_channels=2 * d_model)
        self.up3 = UpSample(in_channels=2 * d_model, out_channels=d_model)

        # 输入和输出线性层
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        pre_x = x  # 保存输入用于残差连接

        # 输入投影并分为两部分
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Transformer at each scale with Downsampling
        print("xn",x.shape)
        x1 = self.transformer1(x)
        x2 = self.down_1(x1)
        x2 = self.transformer2(x2)
        x3 = self.down_2(x2)
        x3 = self.transformer3(x3)
        x4 = self.down_3(x3)
        x4 = self.transformer4(x4)
        print("1",x3.shape)
        print("1",self.up1(x4, target_size=x3.shape[1]).shape)
        # Upsampling and fusion
        x3_up = self.up1(x4, target_size=x3.shape[1]) + x3
        x2_up = self.up2(x3_up, target_size=x2.shape[1]) + x2
        x1_up = self.up3(x2_up, target_size=x1.shape[1]) + x1

        # 融合残差路径并输出
        z = F.silu(z)  # 激活
        x1_up = x1_up * z
        out = self.out_proj(x1_up + pre_x)
        return out


if __name__ == "__main__":
    seq_len, d_model = 128, 64
    model = PyramidTransformer(seq_len=seq_len, d_model=d_model, depth=2, heads=8, dim_head=32, mlp_dim=128, dropout=0.1)

    x = torch.randn(16, seq_len, d_model)  # 输入 (batch_size, seq_len, d_model)
    out = model(x)
    print(out.shape)  # 输出形状 (16, seq_len, d_model)
