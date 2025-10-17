import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, dropout, inner_dim):
        super().__init__()
        self.LN = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out1 = self.LN(x)
        out2 = self.mlp(out1)
        return out2 + x


class MSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.FF = FeedForward(dim, dropout, inner_dim)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x

        return self.FF(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            MSSA(dim, heads, dim_head, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, input_channels, dim, depth, heads, dim_head, dropout):
        """
        光谱特征提取模块（仅基于光谱信息）。
        """
        super(SpectralFeatureExtractor, self).__init__()

        # 光谱加权提取模块 (SWETM)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv1x7 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv1d(128, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        # Transformer 模块
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, dim))  # 假设最大序列长度为 1024
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout)

    def forward(self, x):
        """
        前向传播
        :param x: 输入光谱数据，形状为 (batch_size, channels, spectrum_length)
        :return: 提取的光谱特征，形状为 (batch_size, dim)
        """
        # 光谱卷积模块
        x = self.conv1x1_1(x)
        x = self.conv1x7(x)
        x = self.conv1x1_2(x)

        # 准备输入到 Transformer 的形状
        x = x.permute(0, 2, 1)  # (batch_size, spectrum_length, dim)
        b, n, _ = x.shape

        # 添加类别标记 (CLS Token)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置嵌入
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)

        # 通过 Transformer
        x = self.transformer(x)

        # 返回 CLS Token 的输出
        return x[:, 0]


# 测试模型
def main():
    # 模拟输入
    batch_size = 16
    channels = 1  # 光谱通道数
    spectrum_length = 200  # 光谱长度
    dim = 128  # 特征维度
    depth = 4  # Transformer 层数
    heads = 4  # 多头注意力头数
    dim_head = 32  # 每个头的维度
    dropout = 0.1  # Dropout

    # 创建模型
    model = SpectralFeatureExtractor(channels, dim, depth, heads, dim_head, dropout)

    x = torch.randn(batch_size, channels, spectrum_length)  # 输入光谱数据
    output = model(x)  # 前向传播

    print("输入光谱形状:", x.shape)
    print("输出特征形状:", output.shape)


if __name__ == "__main__":
    main()
