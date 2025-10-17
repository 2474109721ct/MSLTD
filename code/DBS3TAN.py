# -*- coding:utf-8 -*-
from torch import nn                     # 导入 PyTorch 的神经网络模块
from einops import rearrange             # 从 einops 库中导入 rearrange 函数，用于张量的重排操作
import torch                             # 导入 PyTorch 库
import torch.nn.functional as F          # 导入 PyTorch 的函数式 API 并命名为 F
from torch import einsum                 # 从 PyTorch 导入 einsum 函数，用于张量运算
from einops import rearrange, repeat     # 导入 einops 库中的 rearrange 和 repeat 函数
from einops.layers.torch import Rearrange # 导入 einops 的 Rearrange 类，用于定义模型层中的重排操作

# 定义一个辅助函数：若 t 是元组，直接返回；若不是元组，则将其转换为 (t, t) 的形式
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义各个模块的类

# 预归一化模块，包含一个归一化层和一个指定函数
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)    # 定义 LayerNorm 层，对输入进行归一化
        self.fn = fn                     # 存储传入的函数
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) # 对 x 先进行归一化，再应用函数 fn

# 前馈网络模块，包含两层全连接层和激活函数
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(        # 使用 Sequential 封装前馈网络
            nn.Linear(dim, hidden_dim),  # 第一层全连接层，将维度从 dim 变为 hidden_dim
            nn.GELU(),                   # 使用 GELU 激活函数
            nn.Dropout(dropout),         # Dropout 层，防止过拟合
            nn.Linear(hidden_dim, dim),  # 第二层全连接层，维度回到 dim
            nn.Dropout(dropout)          # 再次使用 Dropout
        )
    def forward(self, x):
        return self.net(x)               # 前向传播，通过整个网络

# 自注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads     # 计算注意力层的内部维度
        project_out = not (heads == 1 and dim_head == dim) # 确定是否需要投影输出

        self.heads = heads               # 多头注意力的头数
        self.scale = dim_head ** -0.5    # 缩放因子，用于平衡注意力分数

        self.attend = nn.Softmax(dim = -1) # 定义 softmax，用于计算注意力分数
        self.dropout = nn.Dropout(dropout) # Dropout 层，防止过拟合

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # 线性变换，生成查询 (Q)、键 (K)、值 (V)

        # 如果需要投影输出，则定义线性层；否则使用恒等映射
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 将输入通过 to_qkv 线性层，分割为 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 将 Q, K, V 重排成多头的格式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 计算 Q 和 K 的点积，并缩放
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 计算注意力权重并应用 dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 使用注意力权重加权求和 V
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排回原始维度
        return self.to_out(out)          # 返回输出

# Transformer 模块，包含多层自注意力和前馈网络
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 创建一个空的模块列表，用于存储每一层
        for _ in range(depth):
            # 每层包含一个自注意力模块和一个前馈网络模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x              # 残差连接，应用自注意力模块
            x = ff(x) + x                # 残差连接，应用前馈网络模块
        return x                         # 返回经过 Transformer 处理的输出

# 定义图像分块嵌入模块
class Patch_emd(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, emb_dropout):
        super(Patch_emd, self).__init__()
        image_height, image_width = pair(image_size)   # 获取图像高度和宽度
        patch_height, patch_width = pair(patch_size)   # 获取每个小块的高度和宽度
        # 确保图像尺寸能够被小块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, '图像尺寸必须能被小块尺寸整除'

        patch_dim = channels * patch_height * patch_width # 计算每个小块的维度
        # 定义小块嵌入层，包含一个重排和线性层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 将图像重排为小块
            nn.Linear(patch_dim, dim),    # 线性层将小块投射到指定的维度
        )
        num_patches = (image_height // patch_height) * (image_width // patch_width) # 计算小块数量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     # 定义位置嵌入参数
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 层用于防止过拟合

    def forward(self, img):
        img = self.to_patch_embedding(img)   # 将图像分块并进行线性嵌入
        b, n, _ = img.shape                  # 获取批量大小和小块数量
        img += self.pos_embedding[:, :(n)]   # 添加位置嵌入
        img = self.dropout(img)              # 应用 Dropout
        return img                           # 返回嵌入后的图像

# 定义 ViT（Vision Transformer）类
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)  # 获取图像高度和宽度
        patch_height, patch_width = pair(patch_size)  # 获取每个小块的高度和宽度

        # 检查图像尺寸是否能被小块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, '图像尺寸必须被小块尺寸整除'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 计算小块的数量
        patch_dim = channels * patch_height * patch_width  # 每个小块的维度
        assert pool in {'cls', 'mean'}, '池化方式必须是 cls 或 mean'  # 确保池化类型为 cls 或 mean

        # 小块嵌入层，包含重排和线性层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 将图像重排为小块
            nn.Linear(patch_dim, dim),  # 将小块映射到指定维度
        )

        # 定义位置嵌入和类别标记的参数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding1 = nn.Parameter(torch.randn(1, dim + 1, num_patches))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, num_patches))
        self.dropout = nn.Dropout(emb_dropout)  # Dropout 层，用于防止过拟合

        # 定义两个 Transformer 编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer1 = Transformer(num_patches, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 恒等映射
        self.mlp_head = nn.Sequential(  # 输出头部
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape  # 获取批量大小和小块数量

        # 创建类别标记并拼接到输入
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]  # 添加位置嵌入
        x = self.dropout(x)  # 应用 Dropout
        x = self.transformer(x)  # 通过 Transformer 编码器

        # 进一步处理嵌入
        x1_class = x[:, 0]
        x1 = x[:, 1:]
        x1 = x1.permute(0, 2, 1)  # 转置
        b, n, _ = x1.shape
        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=b)
        x2 = torch.cat((cls_tokens1, x1), dim=1)
        x2 += self.pos_embedding1[:, :(n + 1)]
        x2 = self.dropout(x2)
        x2 = self.transformer1(x2)
        x2_class = x2[:, 0]
        out = torch.cat((x1_class, x2_class), dim=-1)

        out = self.to_latent(out)  # 恒等映射
        return out  # 返回嵌入输出

# 定义 Spa_att（空间注意力）类
class Spa_att(nn.Module):
    def __init__(self):
        super(Spa_att, self).__init__()

    def forward(self, x):
        b, c, h, d = x.size()
        k = torch.clone(x)  # 复制输入，用于计算注意力
        q = torch.clone(x)
        v = torch.clone(x)
        k_hat = rearrange(k, 'b c h d -> b (h d) c')  # 将 k 重新排列
        q_hat = rearrange(q, 'b c h d -> b c (h d)')
        v_hat = rearrange(v, 'b c h d -> b c (h d)')
        k_q = torch.bmm(k_hat, q_hat)  # 计算 K 和 Q 的点积
        k_q = F.softmax(k_q, dim=-1)  # 应用 softmax
        q_v = torch.bmm(v_hat, k_q)  # 计算加权和
        q_v_re = rearrange(q_v, 'b c (h d) -> b c h d', h=h, d=d)
        att = x + q_v_re  # 加上输入实现残差连接
        return att

# 定义主网络 Net 类
class Net(nn.Module):
    def __init__(self, in_cha, patch, num_class):
        super(Net, self).__init__()
        # 定义第一个卷积模块
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(in_cha, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, in_cha, 1, 1),
            nn.BatchNorm2d(in_cha),
            nn.ReLU(inplace=True),
        )
        self.spa_attention = Spa_att()  # 空间注意力模块

        # 定义第二个卷积模块
        self.spa_conv2 = nn.Sequential(
            nn.Conv2d(in_cha, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, in_cha, 1, 1),
            nn.BatchNorm2d(in_cha),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Sequential(nn.Linear(3080, num_class))  # 全连接层

        # 定义 ViT 模块
        self.spe_former = ViT(
            image_size=(patch, patch),
            patch_size=(1, 1),
            num_classes=num_class,
            dim=in_cha,
            depth=2,
            heads=3,
            mlp_dim=256,
            pool='cls',
            channels=in_cha,
            dim_head=64,
            dropout=0.2,
            emb_dropout=0.1,
        )
        self.sigmoid = nn.Sigmoid()  # 定义 Sigmoid 激活函数

    #  这个是the spatial attention module;
    def forward_once(self, x):
        x1 = self.spa_conv1(x)  # 通过第一个卷积模块
        x2 = self.spa_attention(x1)  # 应用空间注意力
        x3 = x + x2  # 残差连接
        x4 = self.spa_conv2(x3)  # 通过第二个卷积模块
        x5 = self.spa_attention(x4)  # 再次应用空间注意力
        x6 = x4 + x5  # 残差连接
        return x6

    def forward_twice(self, x):
        x = self.spe_former(x)  # 通过 ViT 模块
        return x

    def forward_third(self, x):
        x = self.linear(x)  # 通过全连接层
        return x

    def forward(self, x1_spa, x2_spa, x1_band, x2_band):
        x1_spa = self.forward_once(x1_spa)  # 空间分支
        x2_spa = self.forward_once(x2_spa)  # 空间分支

        x1_spe = self.forward_twice(x1_band)  # 频谱分支
        x2_spe = self.forward_twice(x2_band)  # 频谱分支

        # 展平空间特征
        x1_spa = x1_spa.view(x1_spa.shape[0], -1)
        x2_spa = x2_spa.view(x2_spa.shape[0], -1)

        # 计算空间和频谱特征的相似度
        similar1 = F.pairwise_distance(x1_spa, x2_spa)
        similar2 = F.pairwise_distance(x1_spe, x2_spe)

        similar = 0.5 * similar1 + 0.5 * similar2  # 计算整体相似度

        return x1_spa, x2_spa, x1_spe, x2_spe, similar  # 返回特征和相似度

# 定义对比损失类
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 定义对比损失的边界值

    def forward(self, output1, output2, label):
        # 计算欧氏距离
        euclidean_distance = F.pairwise_distance(output1, output2)
        # 计算对比损失
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive  # 返回对比损失值
