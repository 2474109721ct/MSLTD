import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, spec_band):
        """
        多尺度光谱特征提取模块：结合 Conv + Pooling + Transformer
        :param spec_band: 输入光谱维度（光谱波段数量）
        """
        super(SpectralFeatureExtractor, self).__init__()

        self.spec_band = spec_band

        # 定义多尺度卷积层
        self.conv_small = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv_medium = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv_large = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3)

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Transformer 模块
        self.transformer = TransformerModule(dim=96, depth=4, heads=4, dim_head=32, mlp_dim=256)

        # 全连接层
        self.fc = nn.Linear(96, 128)

    def forward(self, x):
        """
        前向传播
        :param x: 输入光谱数据，形状为 (batch_size, spec_band)
        :return: 提取的光谱特征，形状为 (batch_size, 128)
        """
        # 调整输入为 Conv1d 所需的形状 (batch_size, 1, spec_band)
        x = x.unsqueeze(1)

        # 多尺度卷积
        x_small = self.pool(self.conv_small(x))  # 小尺度卷积
        x_medium = self.pool(self.conv_medium(x))  # 中尺度卷积
        x_large = self.pool(self.conv_large(x))  # 大尺度卷积

        # 特征融合
        x = torch.cat([x_small, x_medium, x_large], dim=1)  # 在通道维度上融合
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, seq_len, feature_dim)

        # Transformer
        x = self.transformer(x)

        # 全局池化 + 全连接层
        x = self.fc(x.mean(dim=1))

        return x


class TransformerModule(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        """
        定义 Transformer 模块
        :param dim: 输入特征维度
        :param depth: Transformer 层的深度
        :param heads: 多头注意力的头数
        :param dim_head: 每个头的维度
        :param mlp_dim: 前馈网络隐藏层的维度
        """
        super(TransformerModule, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        前置规范化模块
        """
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        多头自注意力模块
        """
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, self.heads, t.shape[-1] // self.heads).permute(0, 2, 1, 3), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    # 初始化参数
    spec_band = 200  # 假设光谱维度为 200
    batch_size = 16
    epochs = 10
    learning_rate = 1e-4

    # 初始化模型
    model = SpectralFeatureExtractor(spec_band)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 创建假数据
    x_train = torch.randn(batch_size, spec_band)
    y_train = torch.randn(batch_size, 128)

    # 模型训练
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    # 测试模型
    model.eval()
    x_test = torch.randn(batch_size, spec_band)
    with torch.no_grad():
        output = model(x_test)
    print("测试输出形状:", output.shape)


if __name__ == "__main__":
    main()
