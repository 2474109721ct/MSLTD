import torch
import torch.nn as nn
import numpy as np
import random
import scipy.io as sio
import torch.nn.functional as F  # 用于激活函数等操作


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


# 定义specMN_scheme2特征提取模块
class specMN_scheme2(nn.Module):
    def __init__(self, spec_band, output_dim=128, strategy='s1', time_steps=3, init_weights=True):
        super(specMN_scheme2, self).__init__()

        self.strategy = strategy
        self.time_steps = time_steps
        self.spec_band = spec_band

        # 卷积和池化层，用于特征提取和降维
        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, 1, 3, stride=1, padding=1, bias=True).float(),
            nn.MaxPool1d(2, stride=2)
        )

        # 计算分块尺寸
        p0_length = spec_band
        p1_length = p0_length // 2
        p2_length = p1_length // 2
        p3_length = p2_length // 2

        # 根据分块策略 's1' 和 's2' 初始化不同的 LSTM 网络层
        if strategy == 's1':
            self.lstm1 = nn.LSTM(input_size=int(p1_length / 8), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / 4), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / 2), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
        elif strategy == 's2':
            self.lstm1 = nn.LSTM(input_size=int(p1_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm2 = nn.LSTM(input_size=int(p2_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm3 = nn.LSTM(input_size=int(p3_length / time_steps), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
            self.lstm4 = nn.LSTM(input_size=int(p0_length / 1), hidden_size=128, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)

        # 用于融合 LSTM 输出的全连接层，输出检测特征向量
        self.FC = nn.Linear(in_features=128, out_features=output_dim)  # 输出特征向量的维度由output_dim控制

        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec):
        # 将输入重塑以符合卷积层输入格式
        x_spec = x_spec.reshape(-1, 1, x_spec.shape[-1])

        # 逐步通过卷积和池化层
        x1 = self.conv2(x_spec)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)

        # 去掉多余的维度
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)

        # 根据分块策略将池化后的特征数据分块，并送入不同的 LSTM 网络
        if self.strategy == 's1':
            # 分块大小分别为 8、4 和 2
            p1_length = int(x1.shape[-1] / 8)
            p2_length = int(x2.shape[-1] / 4)
            p3_length = int(x3.shape[-1] / 2)
            x1_r = torch.zeros(x_spec.shape[0], 8, p1_length)
            x2_r = torch.zeros(x_spec.shape[0], 4, p2_length)
            x3_r = torch.zeros(x_spec.shape[0], 2, p3_length)

            # 初始化 start 和 end，用于分块x1
            start = 0
            end = min(start + p1_length, x1.shape[-1])
            for i in range(8):
                x1_r[:, i, :] = x1[:, start:end]
                start = end
                end = min(start + p1_length, x1.shape[-1])

            # 初始化 start 和 end，用于分块x2
            start = 0
            end = min(start + p2_length, x2.shape[-1])
            for i in range(4):
                x2_r[:, i, :] = x2[:, start:end]
                start = end
                end = min(start + p2_length, x2.shape[-1])

            # 初始化 start 和 end，用于分块x3
            start = 0
            end = min(start + p3_length, x3.shape[-1])
            for i in range(2):
                x3_r[:, i, :] = x3[:, start:end]
                start = end
                end = min(start + p3_length, x3.shape[-1])


        elif self.strategy == 's2':
            # 分块大小根据时间步长 `time_steps` 计算
            p1_length = int(x1.shape[-1] / self.time_steps)
            p2_length = int(x2.shape[-1] / self.time_steps)
            p3_length = int(x3.shape[-1] / self.time_steps)
            x1_r, x2_r, x3_r = (
                torch.zeros(x_spec.shape[0], self.time_steps, p1_length),
                torch.zeros(x_spec.shape[0], self.time_steps, p2_length),
                torch.zeros(x_spec.shape[0], self.time_steps, p3_length)
            )

            # 填充 x1、x2 和 x3 按策略 's2' 的时间步长规则分块
            for i in range(self.time_steps):
                x1_r[:, i, :] = x1[:, i * p1_length: (i + 1) * p1_length]
                x2_r[:, i, :] = x2[:, i * p2_length: (i + 1) * p2_length]
                x3_r[:, i, :] = x3[:, i * p3_length: (i + 1) * p3_length]

        # 将不同尺度的特征送入 LSTM
        _, (y_1, _) = self.lstm1(x1_r)
        _, (y_2, _) = self.lstm2(x2_r)
        _, (y_3, _) = self.lstm3(x3_r)
        _, (y_4, _) = self.lstm4(x_spec)

        # LSTM 的输出去掉多余维度，输出目标检测的特征向量
        y = F.relu(self.FC(y_1.squeeze(0) + y_2.squeeze(0) + y_3.squeeze(0) + y_4.squeeze(0)))
        return y  # 返回目标检测特征

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


def main():
    # 设置随机种子
    setup_seed(42)

    # 加载和预处理数据
    mat = sio.loadmat('Sandiego.mat')  # 加载 .mat 文件
    data = mat['data']  # 提取数据
    data = standard(data)  # 对数据进行标准化

    # 获取数据的高度、宽度和通道数
    h, w, c = data.shape  # 获取数据的维度信息
    data = np.reshape(data, [-1, c], order='F')  # 将数据展平成 [总像素数, 通道数] 形式

    # 选择一个样本区间
    tp_sample = data[100:110]  # 提取 10 个样本，每个样本有 c 个特征维度

    # 将样本转换为 PyTorch 的 Tensor
    tp_sample_tensor = torch.Tensor(tp_sample)  # 转换为 Tensor

    # 初始化和使用specMN_scheme2模型用于目标检测
    spec_band = c  # 输入的光谱带宽
    output_dim = 128  # 特征输出的维度，用于目标检测
    model = specMN_scheme2(spec_band, output_dim, strategy='s1', time_steps=3)

    # 前向传播，提取特征
    output = model(tp_sample_tensor)  # 输入样本并提取目标检测特征
    print("输出特征的形状:", output.shape)  # 输出特征的形状
    print("提取的特征:\n", output)  # 显示提取的特征


# 执行 main 函数
if __name__ == "__main__":
    main()
