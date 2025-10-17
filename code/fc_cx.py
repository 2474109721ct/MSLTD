import torch
import torch.nn as nn
import numpy as np
import random
import scipy.io as sio
from scipy.sparse import block_diag


# 设置随机种子函数，确保实验可重复
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 数据标准化函数
def standard(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


# 定义带有 mask 的线性层
class EnsembleLinear(nn.Module):
    def __init__(self, input_size, output_size, num_detectors, input_flag=False):
        super(EnsembleLinear, self).__init__()
        mask = tuple([np.ones((output_size, input_size)) for _ in range(num_detectors)])
        mask_array = block_diag(mask).toarray()
        self.linear_mask = nn.Linear(input_size * num_detectors, output_size * num_detectors, bias=False)
        self.mask = torch.Tensor(mask_array)

    def forward(self, x):
        x = torch.matmul(self.linear_mask.weight * self.mask, x.transpose(1, 0))
        return x.transpose(1, 0)


# 全局特征提取模块
class GlobalFeatureExtractor(nn.Module):
    def __init__(self, input_size, num_detectors=4):
        super(GlobalFeatureExtractor, self).__init__()
        self.num_detectors = num_detectors
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size * num_detectors, affine=True),
            EnsembleLinear(input_size, 64, num_detectors),
            nn.BatchNorm1d(64 * num_detectors, affine=True),
            nn.Sigmoid(),
            EnsembleLinear(64, 32, num_detectors),
            nn.BatchNorm1d(32 * num_detectors, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.cat([x for _ in range(self.num_detectors)], dim=-1)
        return self.net(x)


# 局部特征提取模块
class LocalFeatureExtractor(nn.Module):
    def __init__(self, input_size):
        super(LocalFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)  # 一维卷积层
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(8)  # 池化层，用于减小特征维度

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度，(batch_size, 1, input_size)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.flatten(start_dim=1)  # 展平为 (batch_size, 特征维度)


# 综合特征提取模型，将全局和局部特征融合
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, num_detectors=4):
        super(FeatureExtractor, self).__init__()
        self.global_extractor = GlobalFeatureExtractor(input_size, num_detectors)
        self.local_extractor = LocalFeatureExtractor(input_size)

    def forward(self, x):
        global_features = self.global_extractor(x)  # 提取全局特征
        local_features = self.local_extractor(x)  # 提取局部特征
        combined_features = torch.cat([global_features, local_features], dim=1)  # 拼接全局和局部特征
        return combined_features


def main():
    mat = sio.loadmat('Sandiego.mat')
    data = mat['data']
    data = standard(data)
    h, w, c = data.shape
    data = np.reshape(data, [-1, c], order='F')
    tp_sample = data[100:110]
    tp_sample_tensor = torch.Tensor(tp_sample)

    input_size = c
    num_detectors = 4
    model = FeatureExtractor(input_size, num_detectors)

    output = model(tp_sample_tensor)
    print("输出特征的形状:", output.shape)
    print("提取的特征:\n", output)


if __name__ == "__main__":
    main()
