import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
from ts_generation import ts_generation  # 确保这个模块已正确导入
from Tools import standard  # 确保这个模块已正确导入

## 数据集类的定义
class Data(data.Dataset):
    def __init__(self, path):
        mat = sio.loadmat(path)  # 使用 sio.loadmat 函数读取 MATLAB 文件
        data = mat.get('data')  # 使用 .get() 以防止变量名错误
        gt = mat.get('map')  # 获取地面真值数据（用于标注）

        if data is None or gt is None:
            raise ValueError("MATLAB file does not contain 'data' or 'map' variables")

        print("Shape of data before standardization:", data.shape)
        data = standard(data)
        print("Shape of data after standardization:", data.shape)

        # 假设 data 的形状是 (光谱带数, 像素数)
        if len(data.shape) != 2:
            raise ValueError(f"Expected data to have 2 dimensions, but got {len(data.shape)} dimensions")

        # 重新定义 h, w 和 b
        b, pixel_nums = data.shape
        #h = w = int(np.sqrt(pixel_nums))  # 假设是一个正方形图像
      #  if h * w != pixel_nums:
         #   raise ValueError("Pixel count is not a perfect square, cannot reshape into square image")
        h, w=gt.shape
        data = np.reshape(data.T, (h, w, b), order='F')
        gt = np.reshape(gt, (h, w), order='F')  # 确保 gt 的形状与 data 匹配

        ## 获取目标光谱
        target_spectrum = ts_generation(data, gt, 7)  # 得到目标光谱
        #print(target_spectrum)
        ## 将所有像素视为背景像素
        background_samples = np.reshape(data, [-1, b], order='F')

        ## 通过线性表示随机生成目标样本
        alphas = np.random.uniform(0, 0.1, pixel_nums)
        alphas = alphas[:, None]
        print(f"newX: {background_samples.shape}")
        print(f"NEWP: {target_spectrum.shape}")
        print(f"newalphas: {alphas.shape}")
        target_samples = alphas * background_samples + (1 - alphas) * target_spectrum.T
        print(f"target_samples shape: {target_samples.shape}")
        self.target_samples = target_samples
        self.background_samples = background_samples
        self.target_spectrum = target_spectrum.T
        print("self.target_spectrum :", self.target_spectrum.shape)
        self.nums = pixel_nums
        #print(self.background_samples[:10])
        #print("前50个目标样本:")
        #print(self.target_samples[:10])
    ##  用于获取正样本和负样本
    def __getitem__(self, index):
        positive_samples = self.target_samples[index]
        negative_samples = self.background_samples[index]
        #print(f"positive_samples: {positive_samples.shape}")
        #print(f"negative_samples: {negative_samples.shape}")
        #print(positive_samples[:20])
        #print(negative_samples[:20])
        return positive_samples, negative_samples

    def __len__(self):
        return self.nums


if __name__ == '__main__':
    data = Data('urban1.mat')

    # 目标光谱图
    target_spectrum = data.target_spectrum
    plt.figure()
    plt.plot(target_spectrum.T)
    plt.title('Target Spectrum')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()

    # 背景样本光谱图
    background_samples = data.background_samples
    plt.figure()
    plt.plot(background_samples[:100].T)  # 取前100个背景样本绘制
    plt.title('Background Samples Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()

    # 目标样本光谱图
    target_samples = data.target_samples
    plt.figure()
    plt.plot(target_samples[:100].T)  # 取前100个目标样本绘制
    plt.title('Target Samples Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()

    # 正样本和负样本的光谱图
    center, coded_vector = data.__getitem__(128)
    plt.figure()
    plt.plot(center.T, label='Positive Sample')
    plt.plot(coded_vector.T, label='Negative Sample')
    plt.title('Positive and Negative Samples Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
