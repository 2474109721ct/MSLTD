import torch.nn as nn
import torch
import numpy as np
import random
from scipy.sparse import block_diag


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def standard(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


class EnsembleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_detectors, input_flag=False):
        super(EnsembleLinear, self).__init__()
        mask = tuple([np.ones((output_size, input_size)) for _ in range(num_detectors)])
        mask_array = block_diag(mask).toarray()
        self.input_size = input_size
        self.num = num_detectors
        self.output_size = output_size
        self.input = input_flag
        self.linear_mask = nn.Linear(input_size * num_detectors, output_size * num_detectors, bias=False)
        self.mask = torch.Tensor(mask_array)

    def forward(self, x):
        x = torch.matmul(self.linear_mask.weight * self.mask, x.transpose(1, 0))
        x = x.transpose(1, 0)
        return x


class ICLM(nn.Module):
    def __init__(self, channel, r, mmt):
        super(ICLM, self).__init__()
        self.module = nn.BatchNorm1d(channel)
        self.mmt = mmt
        self.r = r

    def forward(self, feat, batchsize=0):
        if batchsize == 0:
            mean_old = self.module.running_mean
            var_old = self.module.running_var
            feat = (feat - mean_old[None]) / ((var_old[None] + 1e-5) ** 0.5)
            return feat
        else:
            prior_feat = feat[:batchsize]
            num_pixel = feat.shape[0] - batchsize
            mean_old = self.module.running_mean
            var_old = self.module.running_var
            ri = int(self.r * num_pixel / prior_feat.shape[0])
            mean_num = feat.shape[0] + prior_feat.shape[0] * ri
            feat_mean = (feat.sum(dim=0) + prior_feat.sum(dim=0) * ri) / mean_num
            feat_val = (((feat - feat_mean[None]) ** 2).sum(dim=0) + ((prior_feat - feat_mean[None]) ** 2).sum(
                dim=0) * ri) / mean_num
            mean_new = (1 - self.mmt) * mean_old + self.mmt * feat_mean
            var_new = (1 - self.mmt) * var_old + self.mmt * feat_val
            self.module.running_mean = mean_new.detach()
            self.module.running_var = var_new.detach()
            feat = (feat - mean_new[None]) / ((var_new[None] + 1e-5) ** 0.5)
        return feat


class FeatureExtractor(nn.Module):
    def __init__(self, input_size, num_detectors=4):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.num_detectors = num_detectors
        setup_seed((torch.rand(1) * 10000).int().item())
        self.net1 = nn.ModuleList()

        num_1 = input_size
        num_2 = 64
        num_3 = 32

        # 增加调试信息，检查每一层的输入和输出
        self.net1 += [
            nn.BatchNorm1d(input_size * num_detectors, affine=True),  # 输入维度 * 检测器数量
            EnsembleLinear(input_size, num_1, num_detectors),
            ICLM(num_1 * num_detectors, r=0.1, mmt=0.8),
            nn.LeakyReLU(0.01),
            EnsembleLinear(num_1, num_2, num_detectors),
            nn.BatchNorm1d(num_2 * num_detectors),
            nn.LeakyReLU(0.01),
            EnsembleLinear(num_2, num_3, num_detectors),
            nn.LeakyReLU(0.01),
        ]

        for l in self.net1.parameters():
            if l.ndim > 1:
                torch.nn.init.kaiming_normal_(l, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 通过调试输出检查输入的形状
        print(f"输入 x 的形状: {x.shape}")
        x = torch.cat([x for _ in range(self.num_detectors)], dim=-1)  # 拼接，创建检测器维度
        print(f"拼接后的 x 形状: {x.shape}")

        # 逐层打印形状以定位尺寸问题
        for i, layer in enumerate(self.net1):
            x = layer(x)
            print(f"层 {i} 的输出形状: {x.shape}")

        # 调整形状以适应输出特征
        x = x.reshape(-1, self.num_detectors * (x.shape[-1] // self.num_detectors))
        print(f"最终输出形状: {x.shape}")
        return x


# 测试 FeatureExtractor 以确保尺寸匹配
if __name__ == "__main__":
    input_size = 204  # 根据您的输入特征数量设置
    num_detectors = 4
    model = FeatureExtractor(input_size, num_detectors)

    # 创建一个测试输入
    test_input = torch.randn(64, input_size)  # 假设 batch_size = 64
    output = model(test_input)
    print("输出特征的形状:", output.shape)
