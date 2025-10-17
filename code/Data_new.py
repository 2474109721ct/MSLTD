import matplotlib.pyplot as plt
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import torch
import cv2  # OpenCV，用于双边滤波
from ts_generation import ts_generation, generate_perturbed_spectra, feature_fusion

# 双边滤波器的实现，用于高光谱图像背景平滑
def bilateral_filter_background(data, d=5, sigma_color=75, sigma_space=75):
    h, w, b = data.shape
    filtered_data = np.zeros_like(data)
    for i in range(b):  # 对每个光谱通道进行双边滤波
        filtered_data[:, :, i] = cv2.bilateralFilter(np.float32(data[:, :, i]), d, sigma_color, sigma_space)
    return filtered_data

# 数据集类的定义，使用双边滤波器进行背景处理
class DataWithSPF2S(data.Dataset):
    def __init__(self, path, epsilon_range=(0.01, 0.05), d=5, sigma_color=75, sigma_space=75):
        # 读取 MATLAB 文件
        mat = sio.loadmat(path)
        data = mat.get('data')  # 加载数据
        gt = mat.get('map')  # 加载地面真值

        if data is None or gt is None:
            raise ValueError("MATLAB file does not contain 'data' or 'map' variables")

        # 标准化数据
        data = (data - np.min(data)) / (np.max(data) - np.min(data))  # 归一化

        # 获取数据形状
        b, pixel_nums = data.shape  # b 是光谱波段数，pixel_nums 是总像素数
        h, w = gt.shape  # h 和 w 是地面真值的高度和宽度
        if h * w != pixel_nums:
            raise ValueError(f"Ground truth size ({h * w}) does not match data size ({pixel_nums})")

        data = np.reshape(data.T, (h, w, b), order='F')
        gt = np.reshape(gt, (h, w), order='F')
        print(f"重塑后的数据形状: {data.shape}")
        print(f"重塑后的地面真值形状: {gt.shape}")
        # 提取背景样本
        background_indices = np.where(gt == 0)
        raw_background_samples = data[background_indices]  # 原始背景样本的高光谱数据
        print(f"Background sample size: {raw_background_samples.shape}")  # 输出背景样本的大小进行检查

        # 确保背景样本能进行滤波处理，不再reshape为固定形状
        background_samples_filtered = bilateral_filter_background(data, d, sigma_color, sigma_space)
        background_samples = background_samples_filtered.reshape(-1, b)  # 经过双边滤波后的背景样本

        # 获取背景样本数量
        num_background_samples = background_samples.shape[0]

        # 生成目标光谱
        self.target_spectrum = ts_generation(data, gt, 7)  # 保存目标光谱
        print(f"重塑后的目标真值形状: {self.target_spectrum.shape}")
        # 生成与背景样本数相同的目标样本（通过扰动目标光谱生成）
        #perturbed_samples = generate_perturbed_spectra(self.target_spectrum, num_background_samples, epsilon_range)

        # 通过特征融合生成最终的目标样本
        fused_samples = []
        for i in range(num_background_samples):
            perturbed_sample_set = generate_perturbed_spectra(self.target_spectrum, num_samples=5, epsilon_range=epsilon_range)
            fused_sample = feature_fusion(perturbed_sample_set)
            fused_samples.append(fused_sample)

        # 将结果保存为2D数组，确保目标样本与背景样本数量一致
        self.target_samples = np.array(fused_samples).reshape(-1, b)  # 确保目标样本为 2D 形状
        self.background_samples = background_samples
        self.target_spectrum = self.target_spectrum.T
        self.nums = background_samples.shape[0]

    def __getitem__(self, index):
        # 获取正样本和负样本
        positive_sample = self.target_samples[index]
        negative_sample = self.background_samples[index]
        return positive_sample, negative_sample

    def __len__(self):
        return self.nums

if __name__ == '__main__':
    # 使用SPF2S方法生成目标样本，背景通过双边滤波器处理
    data = DataWithSPF2S('urban1.mat', epsilon_range=(0.01, 0.05), d=5, sigma_color=75, sigma_space=75)

    # 绘制目标光谱图
    target_spectrum = np.mean(data.target_samples, axis=0)
    plt.figure()
    plt.plot(target_spectrum)
    plt.title('Target Spectrum with Perturbation and Feature Fusion')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()

    # 绘制目标样本的所有光谱曲线
    plt.figure()
    num_samples_to_plot = min(100, data.target_samples.shape[0])  # 确保不会超出样本数量
    for i in range(num_samples_to_plot):  # 绘制前100个目标样本的光谱
        plt.plot(data.target_samples[i, :], label=f'Sample {i}' if i == 0 else "")
    plt.title('All Target Samples Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend(loc="upper right")
    plt.show()

    # 背景样本光谱图
    background_samples = data.background_samples
    plt.figure()
    plt.plot(background_samples[:100].T)  # 绘制前100个背景样本
    plt.title('Background Samples Spectra (After Bilateral Filtering)')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()

    # 绘制正样本和负样本
    positive_sample, negative_sample = data.__getitem__(128)
    plt.figure()
    plt.plot(positive_sample, label='Positive Sample')
    plt.plot(negative_sample, label='Negative Sample')
    plt.title('Positive and Negative Samples Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
