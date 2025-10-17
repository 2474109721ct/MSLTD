from typing import Dict
import torch
from data_genetrate import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from spe import SpectralFeatureExtractor  # 使用 spe.py 中的光谱特征提取模型
import os
from Tools import checkFile, standard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ts_generation import ts_generation
from sklearn import metrics
import random

def seed_torch(seed=1):
    '''
    Fix the random seed for reproducibility
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosin_similarity(x, y):
    x_norm = torch.sqrt(torch.sum(x ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y ** 2, dim=1))
    x_y_dot = torch.sum(torch.multiply(x, y), dim=1)
    return x_y_dot / (x_norm * y_norm + 1e-8)


def cosin_similarity_numpy(x, y):
    assert x.shape[1] == y.shape[1]
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=1))
    x_y = np.sum(np.multiply(x, y), axis=1)
    return x_y / (x_norm * y_norm + 1e-8)


def isia_loss(x, batch_size, margin=1.0, lambd=1):
    '''
    Calculate the loss for intercategory separation and intracategory aggregation
    '''
    positive, negative, prior = x[:batch_size], x[batch_size:2 * batch_size], x[2 * batch_size:]
    p_sim = cosin_similarity(positive, prior)
    n_sim1 = cosin_similarity(negative, prior)
    n_sim2 = cosin_similarity(negative, positive)
    max_n_sim = torch.maximum(n_sim1, n_sim2)

    # Triplet loss
    triplet_loss = margin + max_n_sim - p_sim
    triplet_loss = torch.relu(triplet_loss)
    triplet_loss = torch.mean(triplet_loss)

    # Binary cross-entropy loss
    p_sim = torch.sigmoid(p_sim)
    n_sim = torch.sigmoid(1 - n_sim1)
    bce_loss = -0.5 * torch.mean(torch.log(p_sim + 1e-8) + torch.log(n_sim + 1e-8))

    isia_loss = triplet_loss + lambd * bce_loss
    #isia_loss = triplet_loss
    return isia_loss


def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('Loss Trend')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()

def train(modelConfig: Dict):
    """
    训练函数，基于 SpectralFeatureExtractor 光谱特征提取模型。
    """
    # 设置随机种子
    seed_torch(modelConfig['seed'])

    # 选择设备，默认为 CPU
    device = torch.device("cpu")

    # 数据加载与预处理
    dataset = Data(modelConfig["path"])
    dataloader = DataLoader(
        dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )

    # 模型初始化
    input_dim = 205  # 输入特征维度
    hidden_dims = [256, 128, 64, 32]  # 隐藏层维度
    spec_band = input_dim  # 光谱维度
    net_model = SpectralFeatureExtractor(input_dim, hidden_dims, spec_band).to(device)

    # 如果需要加载预训练权重
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(
                os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]),
                map_location=device
            ),
            strict=False
        )
        print("Model weights loaded.")

    # 优化器与学习率调度器
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )

    # 模型保存路径
    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])
    checkFile(path)

    # 开始训练
    net_model.train()
    loss_list = []

    for e in range(modelConfig["epoch"]):
        epoch_loss = []

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                # 数据组合，生成正样本、负样本和目标光谱
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum.T], axis=0)
                print("combined_vectors:", combined_vectors.shape)

                # 转换为 Tensor 并传递到设备
                combined_vectors = torch.from_numpy(combined_vectors).float().to(device)

                optimizer.zero_grad()  # 梯度清零

                # 前向传播
                features = net_model(combined_vectors)
                print("features:", features.shape)

                # 计算损失
                loss = isia_loss(features, modelConfig['batch_size'])
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"]
                )
                optimizer.step()  # 参数更新

                epoch_loss.append(loss.item())  # 记录当前 batch 的损失

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "batch_loss": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        # 计算并记录当前 epoch 的平均损失
        avg_epoch_loss = np.mean(epoch_loss)
        print(f"Epoch {e}: Average Loss = {avg_epoch_loss}")
        loss_list.append(avg_epoch_loss)

        # 更新学习率
        warmUpScheduler.step()

        # 保存模型权重
        torch.save(net_model.state_dict(), os.path.join(path, f'ckpt_{e}_.pt'))

    # 绘制损失趋势图
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def eval(modelConfig: Dict):
    """
    使用 SpectralFeatureExtractor 模型进行评估。
    """
    # 设置随机种子
    seed_torch(modelConfig['seed'])
    device = torch.device("cpu")
    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])

    with torch.no_grad():
        # 加载数据并进行标准化
        mat = sio.loadmat(modelConfig["path"])
        data = mat['data']
        gt = mat['map']
        data = standard(data)  # 数据标准化
        data = np.float32(data)  # 转换为 float32 类型
        h, w = gt.shape  # 获取图像高度和宽度
        c = data.shape[0]  # 获取光谱维度（通道数）

        # 检查数据大小是否匹配
        if c * h * w != np.prod(data.shape):
            raise ValueError("数据大小与预期的维度不匹配")
        data = np.reshape(data.T, (h, w, c), order='F')  # 重塑数据
        target_spectrum = ts_generation(data, gt, 7)  # 生成目标光谱
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')  # 将数据展平为二维矩阵

        # 初始化模型
        input_dim = c
        hidden_dims = [256, 128, 64, 32]
        spec_band = c
        model = SpectralFeatureExtractor(input_dim, hidden_dims, spec_band)

        # 加载预训练模型权重
        ckpt = torch.load(os.path.join(path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("测试时加载了模型权重。")
        model.eval()  # 设置为评估模式

        detection_map = np.zeros([numpixel])  # 初始化检测结果图

        # 获取目标光谱特征
        target_features = model(
            torch.tensor(target_spectrum.T, dtype=torch.float32).to(device)
        ).cpu().detach().numpy()

        # 逐批处理像素数据
        batch_size = modelConfig['batch_size']
        for i in range(0, numpixel - batch_size, batch_size):
            pixels = torch.tensor(data_matrix[i:i + batch_size], dtype=torch.float32).to(device)
            features = model(pixels).cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

        # 处理剩余像素
        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = torch.tensor(data_matrix[-left_num:], dtype=torch.float32).to(device)
            features = model(pixels).cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

        # 将检测结果重新调整为图像形状
        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)

        # 可视化结果
        false_color_image = generate_false_color_image(data)
        visualize_results(false_color_image, gt, detection_map)

        # 计算 AUC
        y_l = np.reshape(gt, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        auc = round(metrics.auc(fpr, tpr), modelConfig['epsilon'])
        print(f"AUC: {auc:.{modelConfig['epsilon']}f}")

        # 绘制 ROC 曲线
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        # 绘制可分离性图
        separability = tpr - fpr
        threshold = np.linspace(0, 1, num=len(separability))
        plt.figure()
        plt.plot(threshold, separability, marker='.')
        plt.title("Separability Map")
        plt.xlabel("Threshold")
        plt.ylabel("Separability (TPR - FPR)")
        plt.show()


# Helper functions
def generate_false_color_image(data):
    r, g, b = data[:, :, 30], data[:, :, 60], data[:, :, 90]
    r, g, b = (r - r.min()) / (r.max() - r.min()), (g - g.min()) / (g.max() - g.min()), (b - b.min()) / (
            b.max() - b.min())
    return np.dstack((r, g, b))


def visualize_results(false_color_image, gt, detection_result):
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(false_color_image)
    plt.title("False Color Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(detection_result, cmap='hot')
    plt.title("Detection Result")
    plt.axis("off")
    plt.show()
