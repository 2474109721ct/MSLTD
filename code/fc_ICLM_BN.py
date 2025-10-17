import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch  # 导入 PyTorch 主模块
import numpy as np  # 导入 NumPy 模块
import random  # 导入 Python 随机模块
import scipy.io as sio  # 用于加载 .mat 文件
from scipy.sparse import block_diag  # 用于创建 block-diagonal 矩阵

# 设置随机种子函数，确保实验可重复
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 随机种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    random.seed(seed)  # 设置 Python 随机种子
    torch.backends.cudnn.deterministic = True  # 设置 PyTorch 的 cuDNN 后端为确定性

# 数据标准化函数
def standard(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std  # 标准化数据，使其均值为0，方差为1

# 定义对比归一化模块 (CNM)
class CNM(nn.Module):
    def __init__(self, channels, contrast_ratio=0.1, momentum=0.8, blend_factor=0.5):
        super(CNM, self).__init__()
        self.bn = nn.BatchNorm1d(channels)  # 标准批归一化层
        self.contrast_ratio = contrast_ratio  # 对比学习的比例因子
        self.momentum = momentum  # 动量参数，用于更新统计量
        self.blend_factor = blend_factor  # BN和ICLM部分输出的融合系数

    def forward(self, x, prior_batchsize=0):
        # 1. 批归一化部分 (BN)
        x_bn = self.bn(x)  # 直接对输入数据进行批归一化

        # 2. 对比性正则部分 (类似ICLM的对比性增强)
        if prior_batchsize > 0:
            prior_feat = x[:prior_batchsize]
            num_pixels = x.shape[0] - prior_batchsize

            # 计算批次内和批次间的对比均值和方差
            contrast_factor = int(self.contrast_ratio * num_pixels / prior_feat.shape[0])
            mean_combined = (x.mean(dim=0) + prior_feat.mean(dim=0) * contrast_factor) / (1 + contrast_factor)
            var_combined = (x.var(dim=0) + prior_feat.var(dim=0) * contrast_factor) / (1 + contrast_factor)

            # 使用动量来更新均值和方差
            mean_new = (1 - self.momentum) * self.bn.running_mean + self.momentum * mean_combined
            var_new = (1 - self.momentum) * self.bn.running_var + self.momentum * var_combined
            self.bn.running_mean = mean_new.detach()
            self.bn.running_var = var_new.detach()

            # 对比性归一化
            x_iclm = (x - mean_new[None]) / ((var_new[None] + 1e-5) ** 0.5)
        else:
            x_iclm = x_bn  # 如果没有先验批次数据，使用BN结果作为ICLM部分输出

        # 3. 融合BN和ICLM输出
        x_out = self.blend_factor * x_bn + (1 - self.blend_factor) * x_iclm
        return x_out

# 定义一个带有 mask 的线性层
class EnsembleLinear(nn.Module):  # 继承自 PyTorch 的 Module 类
    def __init__(self, input_size, output_size, num_detectors, input_flag=False):
        super(EnsembleLinear, self).__init__()  # 调用父类构造器
        mask = tuple([np.ones((output_size, input_size)) for _ in range(num_detectors)])  # 创建 mask 数组
        mask_array = block_diag(mask).toarray()  # 转换为 block-diagonal 矩阵
        self.input_size = input_size  # 输入维度
        self.num = num_detectors  # 检测器数量
        self.output_size = output_size  # 输出维度
        self.input = input_flag  # 输入标志
        self.linear_mask = nn.Linear(input_size * num_detectors, output_size * num_detectors, bias=False)  # 线性层
        self.mask = torch.Tensor(mask_array)  # 将 mask 转换为 Tensor

    def forward(self, x):
        # 将掩码应用于权重并进行矩阵乘法
        x = torch.matmul(self.linear_mask.weight * self.mask, x.transpose(1, 0))
        x = x.transpose(1, 0)  # 转置输出
        return x  # 返回输出

# 定义特征提取网络
class FeatureExtractor(nn.Module):  # 继承自 PyTorch 的 Module 类
    def __init__(self, input_size, num_detectors=4):
        super(FeatureExtractor, self).__init__()  # 调用父类构造器
        self.input_size = input_size  # 输入维度
        self.num_detectors = num_detectors  # 检测器数量
        setup_seed((torch.rand(1) * 10000).int().item())  # 设置随机种子
        self.net1 = nn.ModuleList()  # 用于存储网络层的列表

        num_1 = input_size  # 第一层输出维度
        num_2 = 32  # 第二层输出维度

        # 构建网络结构，使用CNM替代BN
        self.net1 += [
            CNM(input_size * num_detectors, contrast_ratio=0.1, momentum=0.8, blend_factor=0.5),  # 使用CNM
            EnsembleLinear(input_size, num_1, num_detectors),  # 带有 mask 的线性层
            CNM(num_1 * num_detectors, contrast_ratio=0.1, momentum=0.8, blend_factor=0.5),  # 使用CNM
            nn.Sigmoid(),  # Sigmoid 激活函数
            EnsembleLinear(num_1, num_2, num_detectors),  # 带有 mask 的线性层
            CNM(num_2 * num_detectors, contrast_ratio=0.1, momentum=0.8, blend_factor=0.5),  # 使用CNM
            nn.Sigmoid(),  # Sigmoid 激活函数
        ]

        # 初始化网络参数
        for l in self.net1.parameters():  # 遍历网络参数
            if l.ndim > 1:  # 如果参数维度大于 1，则初始化权重
                torch.nn.init.normal_(l, mean=0, std=0.0001)  # 使用正态分布初始化权重

    def forward(self, x):
        x = torch.cat([x for _ in range(self.num_detectors)], dim=-1)  # 复制输入以适应多个检测器
        for layer in self.net1:  # 遍历网络层
            x = layer(x)  # 逐层处理特征
        x = x.reshape(-1, self.num_detectors * (x.shape[-1] // self.num_detectors))  # 将检测器的特征展平
        return x  # 返回合并后的特征

def main():
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

    # 初始化和使用特征提取模型
    input_size = c  # 输入的特征维度
    num_detectors = 4  # 检测器数量
    model = FeatureExtractor(input_size, num_detectors)

    # 前向传播，提取特征
    output = model(tp_sample_tensor)  # 输入样本并提取特征
    print("输出特征的形状:", output.shape)  # 输出特征的形状
    print("提取的特征:\n", output)  # 显示提取的特征

# 执行 main 函数
if __name__ == "__main__":
    main()
