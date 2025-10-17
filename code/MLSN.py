import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from scipy import io
from torchsummary import summary

# 定义一个1x3卷积函数
def conv1x3(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1)

# 平均池化函数，沿最后一维求平均值
def avgpool(x):
    out = torch.mean(x, dim=2)
    out = out.unsqueeze(1)
    return out

# 定义一个残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = conv1x3(in_channels, out_channels)  # 第一层1x3卷积
        self.conv2 = conv1x3(out_channels, out_channels)  # 第二层1x3卷积
        self.extra = nn.Sequential()  # 额外的1x1卷积，用于匹配通道数
        if out_channels != in_channels:
            self.extra = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            )

    # 前向传播
    def forward(self, x):
        out = F.relu(self.conv1(x))  # 通过第一个卷积层并激活
        out = self.conv2(out)  # 通过第二个卷积层
        out = self.extra(x) + out  # 残差连接
        return out

# 定义另一个残差块，用于特殊用途
class Resone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resone, self).__init__()
        self.conv1 = conv1x3(in_channels, out_channels)  # 单层1x3卷积
        self.extra = nn.Sequential()  # 匹配通道数的1x1卷积
        if out_channels != in_channels:
            self.extra = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            )

    # 前向传播
    def forward(self, x):
        out = self.conv1(x)
        out = self.extra(x) + out  # 残差连接
        return out

# 定义深度残差卷积网络FECNN
class FECNN(nn.Module):
    def __init__(self):
        super(FECNN, self).__init__()
        self.conv1 = conv1x3(1, 40)  # 输入通道为1，输出通道为40
        self.blk1 = Resone(40, 40)  # 残差块1
        self.pool1 = conv1x3(40, 40, 2)  # 下采样1
        self.blk2 = ResBlock(40, 40)  # 残差块2
        self.pool2 = conv1x3(40, 40, 2)  # 下采样2
        self.blk3 = ResBlock(40, 40)  # 残差块3
        self.pool3 = conv1x3(40, 40, 2)  # 下采样3
        self.blk4 = Resone(40, 40)  # 残差块4
        self.pool4 = conv1x3(40, 40, 2)  # 下采样4
        self.pool5 = nn.Conv1d(40, 1, 1)  # 1x1卷积层，用于通道降维
        self.fc = nn.Linear(12, 12)  # 全连接层，用于生成最终的特征向量

    # 前向传播
    def forward(self, x):
        print("x:", x.shape)
        x = F.relu(self.conv1(x))
        print("x:", x.shape)
        x = F.relu(self.blk1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.blk2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.blk3(x))
        x = F.relu(self.pool3(x))
        x = F.relu(self.blk4(x))
        x = F.relu(self.pool4(x))
        x = F.relu(self.pool5(x))
        x = x.flatten(1)  # 展平
        print("x4:", x.shape)
        x = self.fc(x)
        return x

# 定义三元网络，用于三通道嵌入
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    # 前向传播
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    # 获取嵌入
    def get_embedding(self, x):
        return self.embedding_net(x)

# 设定计算设备
device = torch.device('cpu')  # 使用CPU计算

# 主程序
if __name__ == '__main__':
    # 导入 scipy.io 以读取 .mat 文件
    import scipy.io as sio
    # 导入数据标准化工具
    from Tools import standard

    # 读取 .mat 文件中的数据
    mat = sio.loadmat('Sandiego.mat')
    data = mat['data']  # 提取数据
    # 数据标准化
    data = standard(data)  # 对数据进行标准化处理
    # 获取数据的高度、宽度和通道数
    h, w, c = data.shape  # 获取数据的维度信息
    # 重塑数据以便于处理，将其展平为 [总像素数, 通道数] 形式
    data = np.reshape(data, [-1, c], order='F')  # 将数据展平成 2D 数组
    # 选择一个样本区间
    tp_sample = data[100:110]  # 提取10个样本，形状为 [10, 通道数]

    # 将 NumPy 数组转换为 PyTorch 张量并确保数据类型为浮点型
    group_spectra = torch.from_numpy(tp_sample).float()  # 将样本转换为浮点型张量

    # 创建模型实例（假设模型名称为 SpectralGroupAttention）
    model = FECNN()  # 设置 m=1，不进行频谱分组
    # 通过模型提取特征
    features = model(group_spectra.unsqueeze(1))  # 增加维度以适配模型的输入需求
    # 输出特征张量的形状
    print(features.shape)
