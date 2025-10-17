import torch
from torch import nn
from einops import rearrange
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

    def forward(self, out):
        out1 = self.LN(out)  # Norm
        out2 = self.mlp(out1)  # MLP
        out2 = out2 + out1
        return out2


class MSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.FF = FeedForward(dim, dropout, inner_dim)

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, w)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out + x
        out = self.FF(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            MSSA(dim, heads, dim_head, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, channels, dim, depth, heads, dim_head, dropout):
        """
        光谱特征提取模块：结合卷积与 Transformer。
        """
        super(SpectralFeatureExtractor, self).__init__()

        # 卷积层：1D卷积用于提取光谱特征
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(dim)

        # Transformer 模块
        self.transformer = Transformer(dim, heads, dim_head, dropout, depth)

    def forward(self, x):
        """
        前向传播
        :param x: 输入光谱数据，形状为 (batch_size, channels, spectrum_length)
        :return: 提取的光谱特征，形状为 (batch_size, dim)
        """
        # 1D 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Transformer
        x = x.permute(0, 2, 1)  # (batch_size, spectrum_length, dim)
        x = self.transformer(x)

        # 全局平均池化
        x = x.mean(dim=1)
        print(x)
        return x


def main():
    # 测试模型
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

    # 模拟输入
    x = torch.randn(batch_size, channels, spectrum_length)

    # 前向传播
    output = model(x)
    print("输入光谱形状:", x.shape)
    print("输出特征形状:", output.shape)


if __name__ == "__main__":
    main()
