import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch.utils.data as data
import random
import time
import argparse
import traceback
from ts_generation import ts_generation
from Tools import standard

# 软阈值函数
def soft(X, lamd):
    '''
    软阈值函数，应用于信号处理等领域
    :param X: 输入
    :param lamd: 阈值
    :return: 处理后的值
    '''
    t1 = np.sign(X)
    t2 = np.abs(X) - lamd
    t2[t2 < 0] = 0
    return t1 * t2

# SAM 检测器
def SAMDetector(X, P):
    norm_X = np.sqrt(np.sum(np.square(X), axis=0))
    norm_P = np.sqrt(np.sum(np.square(P), axis=0))
    x_dot_P = np.sum(X * P, axis=0)
    value = x_dot_P / (norm_X * norm_P)
    angle = np.arccos(np.clip(value, -1, 1))
    return angle * 180 / np.pi

# 构建背景和目标字典
def dictionaryConstruction(X, p, Kt, m=25, n=5):
    H, W, L = X.shape
    X = np.reshape(X, (H * W, L)).T  # 现在 X 的形状为 (L, H * W)
    detection_result = SAMDetector(X, p)  # SAM 检测结果

    ind = np.argsort(detection_result)
    ind = ind.flatten(order='F')
    ind = ind[:Kt]
    At = X.T[ind]
    target_map = np.zeros([X.shape[1]])
    target_map[ind] = 1
    X = X.T[target_map == 0]  # 获取背景样本
    background_samples = X

    estimator = KMeans(n_clusters=m)
    estimator.fit(X)
    idx = estimator.labels_
    N = np.zeros(shape=[m], dtype=np.int32)
    for i in range(m):
        N[i] = len(np.where(idx == i)[0])
    Xmeans = np.zeros(shape=[m, L], dtype=np.float32)
    for i in range(m):
        Xmeans[i, :] = np.mean(X[idx == i], axis=0)

    Ab = []
    R = []
    for i in range(m):
        if N[i] < L:  #
            continue
        cind = np.where(idx == i)[0]
        Xi = X[cind]
        rXi = Xi - Xmeans[i, :]
        cov = np.matmul(rXi.T, rXi) / (N[i] - 1)
        incov = np.linalg.inv(cov)

        for j in range(N[i]):
            mdj = rXi[j, :].dot(incov).dot(rXi[j, :].T)
            R.append(mdj)

        ind = np.argsort(R)

        Ab.append(X[cind[ind[:n]]])
        R.clear()

    Ab = np.concatenate(Ab, axis=0)
    print(f"newAb: {Ab.shape}")

    return Ab.T, At.T, background_samples

# 独立出来的生成目标样本的函数
def generate_target_samples(Ab, target_spectrum, background_samples):
    '''
    生成目标样本，并确保数量与背景样本一致
    :param Ab: 背景字典
    :param target_spectrum: 目标光谱
    :param background_samples: 背景样本
    :return: 目标样本
    '''
    num_background_samples = background_samples.shape[0]  # 获取背景样本的数量
    N = num_background_samples  # 让目标样本数量等于背景样本数量

    # 生成与背景样本数量相同的 alphas
    alphas = np.random.uniform(0, 0.1, N)[:, None]  # (N, 1)

    # 确保 Ab 的大小与背景样本数匹配
    if Ab.shape[1] < N:
        additional_samples = []
        while len(additional_samples) < N - Ab.shape[1]:
            idx1, idx2 = np.random.choice(Ab.shape[1], 2, replace=False)
            alpha = np.random.uniform(0, 1)
            new_sample = alpha * Ab[:, idx1] + (1 - alpha) * Ab[:, idx2]
            additional_samples.append(new_sample)
        additional_samples = np.array(additional_samples).T
        Ab = np.concatenate([Ab, additional_samples], axis=1)

    Ab = Ab[:, :N]

    # 计算目标样本，确保数量与背景样本一致
    target_samples = alphas * Ab.T + (1 - alphas) * target_spectrum.T

    return target_samples

# 定义 Data 类，继承自 PyTorch 的 Dataset
class Data(data.Dataset):
    def __init__(self, path, Kt=20, m=25, n=5):
        '''
        初始化类，读取数据并生成背景样本和目标样本
        :param path: 数据文件路径
        :param Kt: 目标像素数量
        :param m: 背景聚类数量
        :param n: 每个聚类中的像素数量
        '''
        # 读取数据
        mat = sio.loadmat(path)  # 使用 sio.loadmat 函数读取 MATLAB 文件
        data = mat.get('data')  # 使用 .get() 以防止变量名错误
        gt = mat.get('map')  # 获取地面真值数据（用于标注）

        # 读取数据的形状 (h, w, b)
        h, w, b = data.shape  # h 和 w 是高和宽, b 是波段数
        pixel_nums = h * w  # 像素总数

        # 标准化数据 (保持数据形状不变)
        data = standard(data)

        # 检查地面真值的形状是否与数据的 h, w 匹配
        if gt.shape[0] != h or gt.shape[1] != w:
            raise ValueError(f"地面真值大小 ({gt.shape[0]} * {gt.shape[1]}) 与数据大小 ({h} * {w}) 不匹配")

        # 生成目标光谱
        target_spectrum = ts_generation(data, gt, 7)
        print("target_spectrum1:", target_spectrum.shape)

        # 构建背景字典和目标字典
        Ab, At, background_samples = dictionaryConstruction(
            data, target_spectrum, Kt=Kt, m=m, n=n
        )
        print("background_samples:", background_samples.shape)

        # 调用独立的生成目标样本的函数
        target_samples = generate_target_samples(Ab, target_spectrum, background_samples)

        # 保存必要的数据
        self.target_samples = target_samples
        self.background_samples = background_samples
        self.target_spectrum = target_spectrum
        print("self.background_samples :", self.background_samples.shape)
        print("self.target_samples:", self.target_samples.shape)
        self.nums = pixel_nums  # 使用像素总数

    def __getitem__(self, index):
        if index >= len(self.target_samples):
            index = len(self.target_samples) - 1  # 返回最后一个有效索引的样本
        positive_samples = self.target_samples[index]
        negative_samples = self.background_samples[index]
        return positive_samples, negative_samples

    def __len__(self):
        '''
        获取数据集的样本总数
        :return: 样本总数
        '''
        return self.nums

# 绘制光谱曲线
def plot_spectral_curve(data, title, ylabel="反射率", xlabel="波段", max_samples=4000):
    plt.figure(figsize=(10, 6))
    num_samples = min(data.shape[0], max_samples)
    for i in range(num_samples):
        plt.plot(data[i], label=f"样本 {i+1}" if num_samples <= 10 else "", alpha=0.7)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if num_samples <= 10:
        plt.legend(loc="best")
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="高光谱目标检测脚本")
    args = parser.parse_args()

    start = time.perf_counter()

    try:
        # 创建数据类实例并读取数据
        path = 'Sandiego.mat'
        hyperspectral_data = Data(path)  # 使用新的 Data 类

        # 打印目标样本和背景样本的大小
        print(f"Target samples shape: {hyperspectral_data.target_samples.shape}")
        print(f"Background samples shape: {hyperspectral_data.background_samples.shape}")

        # 获取前100个正样本和负样本并绘制光谱曲线
        plot_spectral_curve(hyperspectral_data.target_samples[:4000], "Target Samples", max_samples=4000)
        plot_spectral_curve(hyperspectral_data.background_samples[:4000], "Background Samples", max_samples=4000)

        # 随机选择一个样本索引
        random_index = random.randint(0, len(hyperspectral_data) - 1)
        positive_sample, negative_sample = hyperspectral_data[random_index]

        # 绘制随机选择的正负样本的对比
        plt.figure()
        plt.plot(positive_sample, label='Positive Sample', color='b')
        plt.plot(negative_sample, label='Negative Sample', color='r')
        plt.title(f'Comparison of Positive and Negative Sample (Index {random_index})')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid(False)
        plt.show()

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()

    end = time.perf_counter()
    print(f'运行完成，用时 {end - start:.2f} 秒')
