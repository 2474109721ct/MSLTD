from typing import Dict
import torch
from data_genetrate import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from model_pray_S3FAN_weizhi import SpectralGroupAttention  # 修改为正确的导入路径
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
    assert x.shape[1] == y.shape[1]
    x_norm = torch.sqrt(torch.sum(x ** 2, dim=1))
    y_norm = torch.sqrt(torch.sum(y ** 2, dim=1))
    x_y_dot = torch.sum(torch.multiply(x, y), dim=1)
    return x_y_dot / (x_norm * y_norm + 1e-8)



def cosin_similarity_numpy(x, y):
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=0)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=0)
    assert x.shape[1] == y.shape[1], f"Features must have the same dimension. x.shape: {x.shape}, y.shape: {y.shape}"
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    dot_product = np.dot(x, y.T)
    return (dot_product / (x_norm * y_norm.T + 1e-8)).squeeze()


def isia_loss(features, batch_size, margin=1.0, lambda_center=0, lambda_bce=1):
    '''
    Calculate the loss using triplet loss, center loss, and binary cross-entropy loss.
    '''
    # 将特征划分为正样本、负样本和先验特征
    positive, negative, prior = features[:batch_size], features[batch_size:2 * batch_size], features[2 * batch_size:]

    # 计算正样本和先验特征之间的余弦相似度
    p_sim = cosin_similarity(positive, prior)
    # 计算负样本和先验特征以及负样本和正样本之间的余弦相似度
    n_sim1 = cosin_similarity(negative, prior)
    n_sim2 = cosin_similarity(negative, positive)
    max_n_sim = torch.maximum(n_sim1, n_sim2)

    # 改进的三元组损失
    triplet_loss = torch.relu(margin + max_n_sim - p_sim).mean()

    # 中心损失
    center = torch.mean(positive, dim=0)  # 计算正样本的中心
    center_loss = torch.mean(torch.sum((positive - center) ** 2, dim=1))  # 正样本与中心的距离

    # 二元交叉熵损失
    p_sim = torch.sigmoid(p_sim)
    n_sim = torch.sigmoid(1 - n_sim1)
    bce_loss = -0.5 * (torch.log(p_sim + 1e-8).mean() + torch.log(n_sim + 1e-8).mean())
    #total_loss = triplet_loss +lambda_bce * bce_loss
    # 总损失
    total_loss = triplet_loss + lambda_center * center_loss + lambda_bce * bce_loss
    return total_loss


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
    seed_torch(modelConfig['seed'])
    # 强制使用 CPU
    device = torch.device("cpu")  # 将设备固定为 CPU
    dataset = Data(modelConfig["path"])
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=True)

    # 模型设置
    net_model = SpectralGroupAttention(
        band=modelConfig['band'],
        group_length=modelConfig['group_length'],
        channel_dim=modelConfig['channel'],
        num_layers=modelConfig['layer']
    ).to(device)  # 将模型移动到 CPU

    # 如果指定了预训练权重，则加载
    if modelConfig["training_load_weight"] is not None:
        weight_path = os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"])
        net_model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print("已加载模型权重。")

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)

    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])
    checkFile(path)

    net_model.train()
    loss_list = []

    for e in range(modelConfig["epoch"]):
        epoch_loss = []

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum.T], axis=0)
                combined_vectors = torch.from_numpy(combined_vectors).float().to(device)  # 数据移动到 CPU
                optimizer.zero_grad()
                features = net_model(combined_vectors)
                loss = isia_loss(features, modelConfig['batch_size'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                epoch_loss.append(loss.item())
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "batch_loss": loss.item(),
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

        avg_epoch_loss = np.mean(epoch_loss)
        print(f"Epoch {e}: Average Loss = {avg_epoch_loss}")
        loss_list.append(avg_epoch_loss)
        warmUpScheduler.step()

        torch.save(net_model.state_dict(), os.path.join(path, f'ckpt_{e}_.pt'))

    paintTrend(loss_list, epochs=modelConfig['epoch'], stride=5)

def select_best(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    # 强制使用 CPU
    device = torch.device("cpu")
    opt_epoch = 0
    max_auc = 0
    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])
    for e in range(modelConfig['epoch']):
        with torch.no_grad():
            mat = sio.loadmat(modelConfig["path"])
            data = mat['data']
            gt = mat['map']
            data = standard(data)
            h, w = gt.shape
            data = np.reshape(data.T, (h, w, data.shape[0]), order='F')
            target_spectrum = ts_generation(data, gt, 7)
            numpixel = h * w
            data_matrix = np.reshape(data, [-1, data.shape[2]], order='F')

            model = SpectralGroupAttention(
                band=modelConfig['band'],
                group_length=modelConfig['group_length'],
                channel_dim=modelConfig['channel'],
                num_layers=modelConfig['layer']
            ).to(device)

            ckpt = torch.load(os.path.join(path, f"ckpt_{e}_.pt"), map_location=device)
            model.load_state_dict(ckpt)
            print(f"Model weights loaded for epoch {e}")
            model.eval()

            batch_size = modelConfig['batch_size']
            detection_map = np.zeros([numpixel])
            target_prior = torch.from_numpy(target_spectrum.T).float().to(device)
            target_features = model(target_prior).cpu().detach().numpy()

            for i in range(0, numpixel, batch_size):
                pixels = data_matrix[i:i + batch_size]
                pixels = torch.from_numpy(pixels).float().to(device)
                features = model(pixels).cpu().detach().numpy()
                detection_map[i:i + len(features)] = cosin_similarity_numpy(features, target_features)
            detection_map = np.reshape(detection_map, [h, w], order='F')
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)

            y_l = np.reshape(gt, [-1]).astype(int)
            y_p = np.reshape(detection_map, [-1])

            fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
            auc = round(metrics.auc(fpr[1:], tpr[1:]), modelConfig['epsilon'])
            if auc > max_auc:
                max_auc = auc
                opt_epoch = e

    print("Best AUC:", max_auc)
    print("Best Epoch:", opt_epoch)

def eval(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    # 强制使用 CPU
    device = torch.device("cpu")
    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])

    with torch.no_grad():
        mat = sio.loadmat(modelConfig["path"])
        data = mat['data']
        gt = mat['map']
        data = standard(data)
        data = np.float32(data)
        h, w = gt.shape
        c = data.shape[0]

        if c * h * w != np.prod(data.shape):
            raise ValueError("数据尺寸不匹配")

        data = np.reshape(data.T, (h, w, c), order='F')
        target_spectrum = ts_generation(data, gt, 7)
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')

        model = SpectralGroupAttention(
            band=modelConfig['band'],
            group_length=modelConfig['group_length'],
            channel_dim=modelConfig['channel'],
            num_layers=modelConfig['layer']
        ).to(device)

        ckpt = torch.load(os.path.join(path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("模型权重已加载")
        model.eval()

        batch_size = modelConfig['batch_size']
        detection_map = np.zeros([numpixel])
        target_prior = torch.from_numpy(target_spectrum.T).float().to(device)
        target_features = model(target_prior).cpu().detach().numpy()

        for i in range(0, numpixel, batch_size):
            pixels = data_matrix[i:i + batch_size]
            pixels = torch.from_numpy(pixels).float().to(device)
            features = model(pixels).cpu().detach().numpy()
            detection_map[i:i + len(features)] = cosin_similarity_numpy(features, target_features)
            print(f"features.shape: {features.shape}")
            print(f"target_features.shape: {target_features.shape}")
            print(f"cosin_similarity_numpy(features, target_features).shape: {cosin_similarity_numpy(features, target_features).shape}")

        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)

        y_l = np.reshape(gt, [-1])
        y_p = np.reshape(detection_map, [-1])

        fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        print(f"AUC: {auc:.{modelConfig['epsilon']}f}")

        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        # Calculate separability (TPR - FPR) at each threshold
        separability = tpr - fpr
        # Ensure the threshold length matches the separability length
        threshold = np.linspace(0, 1, num=len(separability))

        # Plot separability map
        plt.figure()
        plt.plot(threshold, separability, marker='.')
        plt.title("Separability Map")
        plt.xlabel("Threshold")
        plt.ylabel("Separability (TPR - FPR)")
        plt.show()

    # Helper functions
def generate_false_color_image(data):
    # 假设使用第 30、60 和 90 波段作为 R、G、B 通道
    r = data[:, :, 30]
    g = data[:, :, 60]
    b = data[:, :, 90]

    # 归一化到 0-1 范围
    r = (r - np.min(r)) / (np.max(r) - np.min(r))
    g = (g - np.min(g)) / (np.max(g) - np.min(g))
    b = (b - np.min(b)) / (np.max(b) - np.min(b))

    false_color_image = np.dstack((r, g, b))
    return false_color_image

def visualize_results(false_color_image, gt, detection_result):
    plt.figure(figsize=(18, 6))

    # 假彩色图像
    plt.subplot(1, 3, 1)
    plt.imshow(false_color_image)
    plt.title("False Color Image")
    plt.axis("off")

    # 地面真值图
    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    # 检测结果图
    plt.subplot(1, 3, 3)
    plt.imshow(detection_result, cmap='hot')
    plt.title("Detection Result")
    plt.axis("off")

    plt.show()
