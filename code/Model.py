import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat

# 设计网络以获取判别性特征

# 预规范化层
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 创建 LayerNorm 层
        self.norm = nn.LayerNorm(dim)
        # 存储函数 fn 以供后续调用
        self.fn = fn

    def forward(self, x, **kwargs):
        # 应用 LayerNorm 并将结果传递给 fn 函数
        return self.fn(self.norm(x), **kwargs)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # 创建一个包含线性层、激活函数 GELU 和另一个线性层的序列
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # 通过网络传递输入
        return self.net(x)

# 多头自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 决定是否需要投影输出层
        project_out = not (heads == 1 and dim_head == dim)

        # 设置头部数量和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 创建 Softmax 层用于注意力权重计算
        self.attend = nn.Softmax(dim=-1)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 创建线性层用于 QKV 投影
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 创建线性层和 Dropout 层组成的序列，用于投影输出；如果不需要则直接返回 Identity
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 对输入 x 应用线性变换得到 QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 分离 QKV 并重新排列维度以便多头运算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算点积注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用 Softmax 和 Dropout 到注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 应用加权值矩阵得到输出
        out = torch.matmul(attn, v)
        # 重新排列维度以匹配输入
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过输出投影层
        return self.to_out(out)

# Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, group_nums):
        super().__init__()
        # 创建 Transformer 层列表
        self.layers = nn.ModuleList([])
        for i in range(1, depth + 1):
            # 创建 Transformer 层，包含注意力层、前馈网络和卷积层
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim)),
                nn.Conv2d(group_nums, group_nums, kernel_size=(1, 2), stride=1, padding=0)
            ]))

    def forward(self, x):
        # 循环遍历每一层
        for attn, ff, cov2D in self.layers:
            # 扩展 x 的维度以进行下一步处理
            prex = torch.unsqueeze(x, dim=-1)
            # 通过注意力层
            x = attn(x) + x
            # 通过前馈网络
            x = ff(x) + x
            # 扩展 x 的维度
            x = torch.unsqueeze(x, dim=-1)
            # 合并 prex 和 x
            union = torch.cat([prex, x], dim=-1)
            # 通过卷积层
            x = cov2D(union)
            # 去除多余的维度
            x = torch.squeeze(x, dim=-1)
        # 返回最终输出
        return x

# 光谱组注意力模块
class SpectralGroupAttention(nn.Module):
    def __init__(self, band=189, m=20, d=128, depth=4, heads=4, dim_head=64, mlp_dim=64, adjust=False):
        super().__init__()
        # 创建线性层和 LeakyReLU 激活层
        self.linear = nn.Sequential(
            nn.Linear(m, d),
            nn.LeakyReLU()
        )
        #print("self.linear:", self.linear)
        # 创建类别标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        #print("self.cls_token:", self.cls_token.shape)
        #print("self.cls_token:", self.cls_token)
        # 创建位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, band + 1, d))
        # 创建 Transformer 模块
        self.transformer = Transformer(dim=d, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                       group_nums=band + 1)
        # 创建调整层或直接返回 Identity
        if adjust:
            self.adjust = nn.Sequential(
                nn.Linear(d, mlp_dim),
                nn.LeakyReLU(),
                nn.Linear(mlp_dim, mlp_dim // 2)
            )
        else:
            self.adjust = nn.Identity()

    def forward(self, x):
        # 通过线性层
        print("x:", x.shape)
        x = self.linear(x)
       # print("x:", x.shape)
        print("x:", x)
        # 获取批次大小、序列长度和特征维度
        b, n, _ = x.shape
        # 重复类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # 添加类别标记
        x = torch.cat((cls_tokens, x), dim=1)
        # 加上位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        # 通过 Transformer 模块
        x = self.transformer(x)
        # 提取类别标记对应的特征
        class_token = x[:, 0]
        # 通过调整层
        features = self.adjust(class_token)
        # 返回最终特征
        return features

if __name__ == '__main__':
    # 导入 scipy.io 以读取 .mat 文件
    import scipy.io as sio
    # 导入数据标准化工具
    from Tools import standard

    # 读取 .mat 文件中的数据
    mat = sio.loadmat('Sandiego.mat')
    data = mat['data']
    # 数据标准化
    data = standard(data)
    # 获取数据的高度、宽度和通道数
    h, w, c = data.shape
    # 重塑数据以方便处理
    data = np.reshape(data, [-1, c], order='F')
    # 选择一个样本区间
    tp_sample = data[100:110]

    # 将光谱划分为 n 个重叠的组
    m = 20
    # 计算填充大小
    pad_size = m // 2
    # 使用对称模式对样本进行填充
    new_sample = np.pad(tp_sample, ((0, 0), (pad_size, pad_size)),
                        mode='symmetric')
    # 初始化组光谱数组
    group_spectra = np.zeros([10, c, m])
    # 循环遍历通道
    for i in range(c):
        # 从填充后的样本中提取组光谱
        group_spectra[:, i, :] = np.squeeze(new_sample[:, i:i + m])

    # 将 NumPy 数组转换为 PyTorch 张量
    group_spectra = torch.from_numpy(group_spectra).float()
    # 创建 SpectralGroupAttention 模型实例
    model = SpectralGroupAttention(band=c, m=m, d=128)
    # 通过模型提取特征
    features = model(group_spectra)
    # 输出特征张量的形状
    print("1",features)