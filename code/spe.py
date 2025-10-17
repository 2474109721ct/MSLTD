import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np


def standard(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


class SpectralModule(nn.Module):
    def __init__(self, spec_band):
        super(SpectralModule, self).__init__()
        self.spec_band = spec_band

        # 定义不同分段的 LSTM 模块
        self.lstm1 = nn.LSTM(input_size=int(spec_band / 8), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=int(spec_band / 4), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=int(spec_band / 2), hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)
        self.lstm4 = nn.LSTM(input_size=spec_band, hidden_size=128, num_layers=1, bias=True,
                             batch_first=True, dropout=0, bidirectional=False)

        self.FC = nn.Linear(in_features=128, out_features=128)

    def forward(self, x_spec):
        print("x1",x_spec.shape)
        if x_spec.shape[-1] != self.spec_band:
            raise ValueError(f"Input shape {x_spec.shape[-1]} does not match spec_band {self.spec_band}")

        d = x_spec.shape[-1]
        p1_length = int(self.spec_band / 8)
        p2_length = int(self.spec_band / 4)
        p3_length = int(self.spec_band / 2)

        x1 = torch.zeros(x_spec.shape[0], 8, p1_length, device=x_spec.device)
        x2 = torch.zeros(x_spec.shape[0], 4, p2_length, device=x_spec.device)
        x3 = torch.zeros(x_spec.shape[0], 2, p3_length, device=x_spec.device)
        x4 = x_spec.reshape(x_spec.shape[0], 1, self.spec_band)

        start = 0
        end = min(start + p1_length, d)
        for i in range(8):
            x1[:, i, :] = x_spec[:, start:end]
            start = end
            end = min(start + p1_length, d)

        start = 0
        end = min(start + p2_length, d)
        for i in range(4):
            x2[:, i, :] = x_spec[:, start:end]
            start = end
            end = min(start + p2_length, d)

        start = 0
        end = min(start + p3_length, d)
        for i in range(2):
            x3[:, i, :] = x_spec[:, start:end]
            start = end
            end = min(start + p3_length, d)

        _, (y_1, _) = self.lstm1(x1)
        _, (y_2, _) = self.lstm2(x2)
        _, (y_3, _) = self.lstm3(x3)
        _, (y_4, _) = self.lstm4(x4)

        y_1 = y_1.squeeze(0)
        y_2 = y_2.squeeze(0)
        y_3 = y_3.squeeze(0)
        y_4 = y_4.squeeze(0)

        y = y_1 + y_2 + y_3 + y_4
        y = F.relu(self.FC(y))

        return y


class SpectralFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, spec_band):
        super(SpectralFeatureExtractor, self).__init__()
        self.spectral_module = SpectralModule(spec_band)
        self.layers = nn.ModuleList()
        prev_dim = 128  # SpectralModule 输出固定为 128

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            self.layers.append(nn.Sigmoid())
            prev_dim = hidden_dim

    def forward(self, x):
        spectral_features = self.spectral_module(x)
        for layer in self.layers:
            spectral_features = layer(spectral_features)
        return spectral_features


# 测试代码
if __name__ == "__main__":
    mat = sio.loadmat('Sandiego.mat')
    data = mat['data']
    data = standard(data)

    h, w, c = data.shape
    data = np.reshape(data, [-1, c], order='F')

    tp_sample = data[100:110]
    test_input = torch.tensor(tp_sample, dtype=torch.float32)
    print("输入形状:", test_input.shape)

    input_dim = c
    hidden_dims = [256, 128, 64, 32]
    spec_band = c

    model = SpectralFeatureExtractor(input_dim, hidden_dims, spec_band)

    output = model(test_input)
    print("输出特征形状:", output.shape)
