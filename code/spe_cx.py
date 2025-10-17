import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import scipy.io as sio

# 设置随机种子函数，确保实验可重复
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 数据标准化函数
def standard(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

# 多头注意力模块
class MultiheadAttention(nn.Module):
    def __init__(self, input_size, num_heads, channels):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_linear = nn.Linear(input_size, channels)
        self.key_linear = nn.Linear(input_size, channels)
        self.value_linear = nn.Linear(input_size, channels)
        self.final_linear = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()
        query = self.query_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(2, 3)) / np.sqrt(x.size(-1) / self.num_heads)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights, value)

        concat_attention = weighted_sum.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        output = self.final_linear(concat_attention)
        add_norm1 = self.norm1(output + x)

        ff_output = self.linear(add_norm1)
        add_norm2 = self.norm2(ff_output + add_norm1)
        return add_norm2

# 改进后的特征提取网络
class ImprovedSpecMN(nn.Module):
    def __init__(self, spec_band, output_dim=128, strategy='s1', time_steps=3, init_weights=True):
        super(ImprovedSpecMN, self).__init__()

        self.strategy = strategy
        self.time_steps = time_steps
        self.spec_band = spec_band

        # 卷积和池化层，用于特征提取和降维
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True),
            nn.MaxPool1d(2, stride=2)
        )

        # 初始化LSTM网络层
        p0_length = spec_band
        p1_length = p0_length // 2
        p2_length = p1_length // 2
        p3_length = p2_length // 2

        if strategy == 's1':
            self.lstm1 = nn.LSTM(input_size=int(p1_length / 8), hidden_size=128, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / 4), hidden_size=128, batch_first=True)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / 2), hidden_size=128, batch_first=True)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, batch_first=True)
        elif strategy == 's2':
            self.lstm1 = nn.LSTM(input_size=int(p1_length / time_steps), hidden_size=128, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / time_steps), hidden_size=128, batch_first=True)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / time_steps), hidden_size=128, batch_first=True)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, batch_first=True)

        # 自适应多头注意力模块
        self.attention = MultiheadAttention(input_size=128, num_heads=4, channels=128)

        # 改进后的全连接层，增加BatchNorm和LeakyReLU激活函数
        self.FC = nn.Sequential(
            nn.Linear(in_features=128, out_features=output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec):
        # 将输入重塑以符合卷积层输入格式
        x_spec = x_spec.reshape(-1, 1, x_spec.shape[-1])

        # 通过卷积层和池化层
        x1 = self.conv2(x_spec)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)

        # 去掉多余的维度
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)

        # 根据分块策略进行分块并输入 LSTM 网络
        if self.strategy == 's1':
            p1_length = int(x1.shape[-1] / 8)
            p2_length = int(x2.shape[-1] / 4)
            p3_length = int(x3.shape[-1] / 2)
            x1_r = torch.zeros(x_spec.shape[0], 8, p1_length)
            x2_r = torch.zeros(x_spec.shape[0], 4, p2_length)
            x3_r = torch.zeros(x_spec.shape[0], 2, p3_length)

            # 初始化 start 和 end
            start, end = 0, p1_length
            for i in range(8):
                x1_r[:, i, :] = x1[:, start:end]
                start = end
                end = min(start + p1_length, x1.shape[-1])

            start, end = 0, p2_length
            for i in range(4):
                x2_r[:, i, :] = x2[:, start:end]
                start = end
                end = min(start + p2_length, x2.shape[-1])

            start, end = 0, p3_length
            for i in range(2):
                x3_r[:, i, :] = x3[:, start:end]
                start = end
                end = min(start + p3_length, x3.shape[-1])

        # 将不同尺度的特征送入 LSTM
        _, (y_1, _) = self.lstm1(x1_r)
        _, (y_2, _) = self.lstm2(x2_r)
        _, (y_3, _) = self.lstm3(x3_r)
        _, (y_4, _) = self.lstm4(x_spec)

        # 自适应注意力融合
        y = torch.stack([y_1.squeeze(0), y_2.squeeze(0), y_3.squeeze(0), y_4.squeeze(0)], dim=1)
        y = self.attention(y)
        y = y.mean(dim=1)  # 平均池化融合各尺度的特征

        # 最终全连接层
        y = self.FC(y)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def main():
    # 设置随机种子
    setup_seed(42)

    # 加载和预处理数据
    mat = sio.loadmat('Sandiego.mat')  # 加载 .mat 文件
    data = mat['data']  # 提取数据
    data = standard(data)  # 对数据进行标准化

    # 获取数据的高度、宽度和通道数
    h, w, c = data.shape
    data = np.reshape(data, [-1, c], order='F')  # 将数据展平成 [总像素数, 通道数] 形式

    # 选择一个样本区间
    tp_sample = data[100:110]  # 提取 10 个样本

    # 将样本转换为 PyTorch 的 Tensor
    tp_sample_tensor = torch.Tensor(tp_sample)

    # 初始化并使用 ImprovedSpecMN 模型
    spec_band = c
    output_dim = 128
    model = ImprovedSpecMN(spec_band, output_dim, strategy='s1', time_steps=3)

    # 前向传播，提取特征
    output = model(tp_sample_tensor)
    print("输出特征的形状:", output.shape)
    print("提取的特征:\n", output)

# 执行 main 函数
if __name__ == "__main__":
    main()
