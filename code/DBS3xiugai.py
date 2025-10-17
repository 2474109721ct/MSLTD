# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange  # 用于张量重排操作


class Spa_att(nn.Module):
    def __init__(self):
        super(Spa_att, self).__init__()

    def forward(self, x):
        # 进行空间注意力机制的计算
        b, c, h, d = x.size()
        k = torch.clone(x)
        q = torch.clone(x)
        v = torch.clone(x)

        # 重新排列 k, q, v 的维度
        k_hat = rearrange(k, 'b c h d -> b (h d) c')
        q_hat = rearrange(q, 'b c h d -> b c (h d)')
        v_hat = rearrange(v, 'b c h d -> b c (h d)')

        # 计算注意力分数矩阵
        k_q = torch.bmm(k_hat, q_hat)
        k_q = F.softmax(k_q, dim=-1)

        # 生成加权后的值
        q_v = torch.bmm(v_hat, k_q)
        q_v_re = rearrange(q_v, 'b c (h d) -> b c h d', h=h, d=d)

        # 将注意力加权后的特征与原始输入相加
        att = x + q_v_re
        return att


class Net(nn.Module):
    def __init__(self, band):
        super(Net, self).__init__()
        # 定义第一个卷积模块
        self.spa_conv1 = nn.Sequential(
            nn.Conv2d(band, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, band, 1, 1),
            nn.BatchNorm2d(band),
            nn.ReLU(inplace=True),
        )
        # 定义空间注意力模块
        self.spa_attention = Spa_att()
        # 定义第二个卷积模块
        self.spa_conv2 = nn.Sequential(
            nn.Conv2d(band, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32),
            nn.Conv2d(64, band, 1, 1),
            nn.BatchNorm2d(band),
            nn.ReLU(inplace=True),
        )

    def forward_once(self, x):
        # 第一次卷积和注意力计算
        print("x :", x.shape)
        x1 = self.spa_conv1(x)
        x2 = self.spa_attention(x1)
        x3 = x + x2
        # 第二次卷积和注意力计算
        x4 = self.spa_conv2(x3)
        x5 = self.spa_attention(x4)
        # 输出最终的特征 x6
        x6 = x4 + x5
        return x6

    def forward(self, x):
        return self.forward_once(x)

