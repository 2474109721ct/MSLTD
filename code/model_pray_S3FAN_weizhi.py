import torch
from torch import nn
from pray_S3FAN_weizhi import PyramidTransformer  # 导入 PyramidTransformer

'''
设计模型以提取区分性特征
'''
class SpectralGroupAttention(nn.Module):
    def __init__(self, band=205, group_length=20, channel_dim=128, num_layers=4):
        super().__init__()
        # 计算步幅和序列长度
        stride = group_length // 4
        assert group_length <= band, "group_length 必须小于或等于 band"
        assert stride > 0, "stride 必须大于 0"

        self.group_division = nn.Sequential(
            nn.Conv1d(1, channel_dim, kernel_size=group_length, stride=stride, padding=0),
            nn.LeakyReLU()
        )
        # 计算卷积后的序列长度
        self.sequence_length = (band - group_length) // stride + 1
        assert self.sequence_length > 0, "卷积后的序列长度必须大于 0"

        # 初始化 PyramidTransformer，根据实际定义传递参数
        self.pyramid_transformer = PyramidTransformer(
            seq_len=self.sequence_length,
            d_model=channel_dim,
            heads=4,  # 示例值，调整为适合您的模型
            num_layers=num_layers,  # Transformer 层数
            dim_head=32,  # 示例值
            mlp_dim=128,  # 示例值
            dropout=0.1  # Dropout 概率
        )

        # 全连接网络层
        self.fnn = nn.Sequential(
            nn.Linear(self.sequence_length * channel_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        # 输入调整为 [batch_size, in_channels, seq_len]
        x = x.unsqueeze(1).to(next(self.parameters()).device)  # 将输入扩展为 (batch_size, 1, band)，并移动到模型设备
        x = self.group_division(x)  # 经过卷积分组 (batch_size, channel_dim, sequence_length)
        x = x.permute(0, 2, 1)  # 调整维度为 (batch_size, sequence_length, channel_dim)
        x = self.pyramid_transformer(x)  # 经过 PyramidTransformer
        x = x.reshape([x.shape[0], -1])  # 展平为 (batch_size, sequence_length * channel_dim)
        #x = self.fnn(x)  # 全连接层处理
        print("1", x.shape)
        return x


if __name__ == '__main__':
    # 检查是否支持 CPU
    device = torch.device("cpu")  # 强制使用 CPU
    print(f"Using device: {device}")

    batch_size = 80  # 设置批量大小为 80
    bands = [189, 189, 205, 102]  # 频带尺寸
    group_lengths = [30, 5, 5, 15]  # 分组长度
    channels = [16, 16, 16, 16]  # 通道维度
    num = 0  # 选择第 0 个配置
    input_size = (bands[num],)  # 定义输入形状（不包括批量维度）

    # 初始化 SpectralGroupAttention 模型
    model = SpectralGroupAttention(
        band=bands[num],
        group_length=group_lengths[num],
        channel_dim=channels[num],
        num_layers=4  # PyramidTransformer 的深度
    ).to(device)  # 将模型移动到 CPU

    from torchsummary import summary
    from thop import profile

    # 调整输入形状并检查模型结构
    summary(model, input_size=(bands[num],), device="cpu")  # 确保使用 CPU 进行总结

    # 使用 thop 计算 FLOPs 和参数量
    input_tensor = torch.randn(batch_size, *input_size).to(device)  # 模拟批量输入并移动到 CPU
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    # 将 FLOPs 和参数量转换为人类可读的格式
    flops_million = flops / 10 ** 9
    params_million = params / 10 ** 6

    print(f'FLOPs: {flops_million:.6f} GFLOPs')
    print(f'Parameters: {params_million:.6f} MParams')
