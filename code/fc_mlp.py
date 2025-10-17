import torch.nn as nn
import torch
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

# 定义一个带有 mask 的线性层
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

# 定义多头注意力机制
class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads, channels):
        super(MultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.channels = channels

        self.query_linear = nn.Linear(input_size, channels)
        self.key_linear = nn.Linear(input_size, channels)
        self.value_linear = nn.Linear(input_size, channels)

        self.final_linear = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        batch_size, sequence_length, input_size = x.size()

        # 计算 query, key 和 value
        query = self.query_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        # 计算注意力分数和权重
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / torch.sqrt(torch.tensor(self.channels / self.num_heads))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights, value)

        # 组合多头注意力
        concat_attention = weighted_sum.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        # 应用最后的线性层和残差连接
        output = self.final_linear(concat_attention)
        add_norm1 = self.norm1(output + x)

        # 前馈网络层和第二次残差连接
        ff = self.linear(add_norm1)
        add_norm2 = self.norm2(ff + add_norm1)

        return add_norm2

# 定义特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, num_detectors=4):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.num_detectors = num_detectors
        setup_seed((torch.rand(1) * 10000).int().item())

        num_1 = input_size
        num_2 = 32

        # 构建网络结构，包含多头注意力机制
        self.net1 = nn.ModuleList([
            nn.BatchNorm1d(input_size * num_detectors, affine=True),
            EnsembleLinear(input_size, num_1, num_detectors),
            nn.BatchNorm1d(num_1 * num_detectors, affine=True),
            nn.Sigmoid()
        ])

        # 多头注意力模块
        self.attention = MultiheadAttention(num_1 * num_detectors, num_heads=4, channels=num_1 * num_detectors)

        # 后续网络层
        self.net2 = nn.ModuleList([
            EnsembleLinear(num_1, num_2, num_detectors),
            nn.BatchNorm1d(num_2 * num_detectors, affine=True),
            nn.Sigmoid()
        ])

        # 初始化网络参数
        for l in self.net1.parameters():
            if l.ndim > 1:
                torch.nn.init.normal_(l, mean=0, std=0.0001)
        for l in self.net2.parameters():
            if l.ndim > 1:
                torch.nn.init.normal_(l, mean=0, std=0.0001)

    def forward(self, x):
        x = torch.cat([x for _ in range(self.num_detectors)], dim=-1)

        # 第一个网络块
        for layer in self.net1:
            x = layer(x)

        # 将输入 reshape 为 (batch_size, sequence_length, input_size) 以适配多头注意力
        x = x.view(-1, self.num_detectors, x.shape[-1] // self.num_detectors)
        x = self.attention(x)

        # 展平多头注意力的输出
        x = x.view(-1, self.num_detectors * (x.shape[-1] // self.num_detectors))

        # 第二个网络块
        for layer in self.net2:
            x = layer(x)

        return x

def main():
    # 加载和预处理数据
    mat = sio.loadmat('Sandiego.mat')
    data = mat['data']
    data = standard(data)

    # 获取数据的高度、宽度和通道数
    h, w, c = data.shape
    data = np.reshape(data, [-1, c], order='F')

    # 选择一个样本区间
    tp_sample = data[100:110]
    tp_sample_tensor = torch.Tensor(tp_sample)

    # 初始化和使用特征提取模型
    input_size = c
    num_detectors = 4
    model = FeatureExtractor(input_size, num_detectors)

    # 前向传播，提取特征
    output = model(tp_sample_tensor)
    print("输出特征的形状:", output.shape)
    print("提取的特征:\n", output)

# 执行 main 函数
if __name__ == "__main__":
    main()
