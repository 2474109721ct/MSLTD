# model_utils.py

import torch.nn as nn

# 定义一个简单的模型用于测试对抗性扰动
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)
