import numpy as np
import torch

# 分光谱成组函数，处理边界时进行对称填充或常量填充
def spectral_group(x, n, m):
    pad_size = m // 2

    # 对数据进行对称填充，确保可以进行完整的分组
    new_sample = np.pad(x, ((0, 0), (pad_size, pad_size)), mode='symmetric')

    b = x.shape[0]  # b 是样本数
    group_spectra = np.zeros([b, n, m])

    # 分组操作，将每个分组的长度固定为 m
    for i in range(n):
        segment = new_sample[:, i:i + m]

        # 如果分段的实际大小小于 m，进行填充
        if segment.shape[1] < m:
            # 如果片段为空或不足，采用常量填充代替对称填充
            padding_size = m - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)

        group_spectra[:, i, :] = segment

    return torch.from_numpy(group_spectra).float()
