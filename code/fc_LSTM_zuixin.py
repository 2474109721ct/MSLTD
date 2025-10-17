import torch.nn as nn
import torch
import numpy as np
import random
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
class ensemble_linear(nn.Module):
    def __init__(self, input_size, output_size, num_detectors):
        super(ensemble_linear, self).__init__()
        mask = tuple([np.ones((output_size, input_size)) for _ in range(num_detectors)])
        mask_array = block_diag(mask).toarray()
        self.input_size = input_size
        self.num_detectors = num_detectors
        self.output_size = output_size
        self.linear_mask = nn.Linear(input_size * num_detectors, output_size * num_detectors, bias=False)
        self.mask = torch.Tensor(mask_array)

    def forward(self, x):
        x = torch.matmul(self.linear_mask.weight * self.mask, x.transpose(1, 0))
        x = x.transpose(1, 0)
        return x


# 定义 Conv + Pooling + LSTM 模块
class ConvLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ConvLSTMLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.hidden_size = hidden_size

    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = x.unsqueeze(1)  # 增加通道维度，变为 (batch_size, 1, seq_len)
        x = self.conv(x)  # 卷积后 (batch_size, 32, seq_len)
        x = self.pool(x)  # 池化后 (batch_size, 32, seq_len / 2)

        # 动态计算 LSTM 的输入维度
        seq_len = x.shape[2]
        x = x.permute(0, 2, 1)  # 调整维度为 (batch_size, seq_len / 2, 32)
        lstm = nn.LSTM(input_size=32, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        _, (hn, _) = lstm(x)  # LSTM 输出 (batch_size, hidden_size)
        return hn.squeeze(0)


# 定义特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, num_detectors=4):
        super(FeatureExtractor, self).__init__()
        self.input_size = input_size
        self.num_detectors = num_detectors
        setup_seed((torch.rand(1) * 10000).int().item())

        # 构建网络结构
        self.net1 = nn.ModuleList([
            nn.BatchNorm1d(input_size, affine=True),  # 批量归一化层
            ConvLSTMLayer(input_size=input_size, hidden_size=128),  # Conv + Pooling + LSTM 模块
            nn.BatchNorm1d(128, affine=True),  # 批量归一化层
            nn.Sigmoid(),  # 激活函数
            ConvLSTMLayer(input_size=128, hidden_size=64),  # 第二层 Conv + Pooling + LSTM 模块
            nn.BatchNorm1d(64, affine=True),  # 批量归一化层
            nn.Sigmoid(),  # 激活函数
            ensemble_linear(64, 32, num_detectors),  # 使用 ensemble_linear 作为输出模块
        ])

    def forward(self, x):
        for i, layer in enumerate(self.net1):
            if isinstance(layer, ConvLSTMLayer):
                x = layer(x)  # 处理 ConvLSTMLayer 层
            else:
                x = layer(x)  # 处理其他层
            print(f"Layer {i} output shape: {x.shape}")
        return x


# 测试代码
def main():
    # 模拟数据
    batch_size = 10
    seq_len = 128  # 输入光谱序列长度
    input_data = torch.randn(batch_size, seq_len)  # 随机生成输入数据

    # 初始化和使用特征提取模型
    model = FeatureExtractor(input_size=seq_len)

    # 前向传播，提取特征
    output = model(input_data)
    print("最终输出的形状:", output.shape)
    print("提取的特征:\n", output)


# 执行 main 函数
if __name__ == "__main__":
    main()
