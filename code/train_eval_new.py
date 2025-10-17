from typing import Dict
import torch
from data_genetrate import Data
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Scheduler import GradualWarmupScheduler
from Siamese_fc import FeatureExtractor  # 使用新的 FECNN 模型
import os
from Tools import checkFile, standard
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from ts_generation import ts_generation
from sklearn import metrics
import random
import time
from torchsummary import summary

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
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig.get("device", "cpu"))
    dataset = Data(modelConfig["path"])
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True,
                            pin_memory=True)

    input_size = 207  # 输入特征维度
    num_detectors = modelConfig.get("num_detectors", 4)
    net_model = FeatureExtractor(input_size=input_size, num_detectors=num_detectors).to(device)

    # Log Model Parameters and FLOPs
    print("Model Summary")
    summary(net_model, input_size=(input_size,), device=str(device))

    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(os.path.join(modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device),
            strict=False)
        print("Model weights loaded.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    path = os.path.join(modelConfig["save_dir"], modelConfig['path'])
    checkFile(path)

    net_model.train()
    loss_list = []
    total_training_time = 0

    for e in range(modelConfig["epoch"]):
        start_time = time.time()
        epoch_loss = []

        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positive, negative in tqdmDataLoader:
                combined_vectors = np.concatenate([positive, negative, dataset.target_spectrum.T], axis=0)
                combined_vectors = torch.from_numpy(combined_vectors).float().to(device)

                optimizer.zero_grad()
                features = net_model(combined_vectors)
                loss = isia_loss(features, modelConfig['batch_size'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
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

        # Save model checkpoint
        torch.save(net_model.state_dict(), os.path.join(
            path, 'ckpt_' + str(e) + "_.pt"))

        end_time = time.time()
        epoch_time = end_time - start_time
        total_training_time += epoch_time
        print(f"Epoch {e} completed in {epoch_time:.2f} seconds.")

    print(f"Total Training Time: {total_training_time:.2f} seconds.")
    paintTrend(loss_list, epochs=modelConfig['epoch'], stride=5)


def select_best(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
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
            data = np.float32(data)
            h, w = gt.shape
            c = data.shape[0]
            if c * h * w != np.prod(data.shape):
                raise ValueError("Data size does not match expected dimensions")
            data = np.reshape(data.T, (h, w, c), order='F')
            target_spectrum = ts_generation(data, gt, 7)
            numpixel = h * w
            data_matrix = np.reshape(data, [-1, c], order='F')
            input_size = 207 # 输入特征维度
            num_detectors = modelConfig.get("num_detectors", 4)  # 获取num_detectors参数
            model = FeatureExtractor(input_size=input_size, num_detectors=num_detectors)
            ckpt = torch.load(os.path.join(path, f"ckpt_{e}_.pt"), map_location=device)
            model.load_state_dict(ckpt)
            print(f"Model weights loaded for epoch {e}")
            model.eval()

            detection_map = np.zeros([numpixel])
            target_features = model(
                torch.from_numpy(target_spectrum.T).float().to(device)).cpu().detach().numpy()

            batch_size = modelConfig['batch_size']
            for i in range(0, numpixel - batch_size, batch_size):
                pixels = torch.from_numpy(data_matrix[i:i + batch_size]).float().to(device)
                features = model(pixels).cpu().detach().numpy()
                detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

            left_num = numpixel % batch_size
            if left_num != 0:
                pixels = torch.from_numpy(data_matrix[-left_num:]).float().to(device)
                features = model(pixels).cpu().detach().numpy()
                detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

            detection_map = np.reshape(detection_map, [h, w], order='F')
            detection_map = standard(detection_map)
            detection_map = np.clip(detection_map, 0, 1)
            # 在计算 ROC 曲线前，确保 y_l 和 y_p 是一维的，并且 y_l 仅包含 0 和 1
            y_l = np.reshape(gt, [-1]).astype(int)  # 确保 y_l 是一维数组并且为整数类型
            y_p = np.reshape(detection_map, [-1])  # 确保 y_p 是一维数组

            # 检查 y_l 是否为二进制标签
            if not np.array_equal(np.unique(y_l), [0, 1]):
                raise ValueError("y_l 需要包含二进制标签（0 和 1）")

            # 计算 ROC 曲线
            fpr, tpr, _ = metrics.roc_curve(y_l, y_p, drop_intermediate=False)

            auc = round(metrics.auc(fpr[1:], tpr[1:]), modelConfig['epsilon'])
            if auc > max_auc:
                max_auc = auc
                opt_epoch = e
    print("Best AUC:", max_auc)
    print("Best Epoch:", opt_epoch)


def eval(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
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
            raise ValueError("Data size does not match expected dimensions")
        data = np.reshape(data.T, (h, w, c), order='F')
        target_spectrum = ts_generation(data, gt, 7)
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')

        input_size = 207
        num_detectors = modelConfig.get("num_detectors", 4)
        model = FeatureExtractor(input_size=input_size, num_detectors=num_detectors).to(device)

        # Log Model Parameters and FLOPs
        print("Model Summary During Evaluation")
        summary(model, input_size=(input_size,), device=str(device))

        ckpt = torch.load(os.path.join(path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("Model weights loaded for testing.")
        model.eval()

        start_time = time.time()
        detection_map = np.zeros([numpixel])
        target_features = model(
            torch.from_numpy(target_spectrum.T).float().to(device)).cpu().detach().numpy()

        batch_size = modelConfig['batch_size']
        for i in range(0, numpixel - batch_size, batch_size):
            pixels = torch.from_numpy(data_matrix[i:i + batch_size]).float().to(device)
            features = model(pixels).cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity_numpy(features, target_features)

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = torch.from_numpy(data_matrix[-left_num:]).float().to(device)
            features = model(pixels).cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity_numpy(features, target_features)

        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")

        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)
        save_detection_result_image(detection_map, modelConfig, epoch=modelConfig["epoch"])
        false_color_image = generate_false_color_image(data)
        false_color_image_path = os.path.join(modelConfig["save_dir"], "false_color_image.png")
        plt.imsave(false_color_image_path, false_color_image)
        print(f"False color image saved at: {false_color_image_path}")

        # Save the ground truth image
        ground_truth_image_path = os.path.join(modelConfig["save_dir"], "ground_truth_image.png")
        plt.imsave(ground_truth_image_path, gt, cmap='viridis')
        print(f"Ground truth image saved at: {ground_truth_image_path}")
        save_detection_result_image(detection_map, modelConfig, epoch=modelConfig["epoch"])
        false_color_image = generate_false_color_image(data)
        visualize_results(false_color_image, gt, detection_map)
        y_l = np.reshape(gt, [-1])
        y_p = np.reshape(detection_map, [-1])
        background_similarity = y_p[y_l == 0]  # 背景相似度
        target_similarity = y_p[y_l == 1]  # 目标相似度

        # 使用箱型图可视化分离性
        visualize_separability_boxplot(target_similarity, background_similarity)
        fpr, tpr, thresholds = metrics.roc_curve(y_l, y_p, drop_intermediate=False)

        # Fixing AUC calculation for PFT and PDT
        thresholds = thresholds[np.isfinite(thresholds)]
        fpr = fpr[:len(thresholds)]
        tpr = tpr[:len(thresholds)]

        # Ensure thresholds are sorted in ascending order
        sorted_indices = np.argsort(thresholds)
        thresholds = thresholds[sorted_indices]
        fpr = fpr[sorted_indices]
        tpr = tpr[sorted_indices]

        # Calculate AUC metrics using trapezoidal rule
        auc_pdpf = round(metrics.auc(fpr, tpr), 5)
        auc_pft = round(metrics.auc(thresholds, fpr), 5)
        auc_pdt = round(metrics.auc(thresholds, tpr), 5)
        auc_oa = round(auc_pdpf + auc_pdt - auc_pft, 5)
        auc_snpr = round(auc_pdt / (auc_pft + 1e-8), 5)  # Avoid division by zero

        # Print AUC metrics
        print(f"AUC_PDPF: {auc_pdpf}")
        print(f"AUC_PFT: {auc_pft}")
        print(f"AUC_PDT: {auc_pdt}")
        print(f"AUC_OA: {auc_oa}")
        print(f"AUC_SNPR: {auc_snpr}")

        # 1. Visualize PDPF Curve (ROC curve) with white background and no grid
        plt.figure(facecolor='white')  # Set the background to white
        plt.plot(fpr, tpr, color='cyan', linewidth=2)  # Curve color changed to red
        plt.xscale("log")
        plt.xlim(1e-4, 1)
        plt.ylim(0, 1.05)
        plt.xlabel("False Positive Rate ($P_F$)")
        plt.ylabel("True Positive Rate ($P_D$)")
        plt.title("AUC_PDPF Curve")
        plt.grid(False)  # Disable grid
        plt.show()  # Display the PDPF curve

        # 2. Visualize PFT Curve with white background and no grid
        plt.figure(facecolor='white')  # Set the background to white
        plt.plot(thresholds, fpr, color='cyan', linewidth=2)  # Curve color changed to red
        plt.xlabel("Threshold ($\tau$)")
        plt.ylabel("False Positive Rate ($P_F$)")
        plt.title("AUC_PFT Curve")
        plt.grid(False)  # Disable grid
        plt.show()  # Display the PFT curve

        # 3. Visualize PDT Curve with white background and no grid
        plt.figure(facecolor='white')  # Set the background to white
        plt.plot(thresholds, tpr, color='cyan', linewidth=2)  # Curve color changed to red
        plt.xlabel("Threshold ($\tau$)")
        plt.ylabel("True Positive Rate ($P_D$)")
        plt.title("AUC_PDT Curve")
        plt.grid(False)  # Disable grid
        plt.show()  # Display the PDT curve

        # 4. Visualize 3D ROC Curve with corrected axis orientation, white background, and black coordinate lines
        fig = plt.figure(figsize=(10, 8))
        fig.patch.set_facecolor('white')  # Set figure background to white
        ax = fig.add_subplot(111, projection='3d')

        # Set the background of the 3D axes to white
        ax.set_facecolor('white')

        # Plot the 3D curve, swapping PF and τ positions
        ax.plot(thresholds, fpr, tpr, color='cyan', linewidth=2, label="3D ROC Curve")

        # Set labels and axis ranges (reversed ranges for τ and PF)
        ax.set_xlabel(r'$\tau$', labelpad=15)  # X-axis: Thresholds (τ)
        ax.set_ylabel('$P_F$', labelpad=15)  # Y-axis: False Positive Rate (PF)
        ax.set_zlabel('$P_D$', labelpad=15)  # Z-axis: True Positive Rate (PD)

        ax.set_xlim(1, 0)  # X-axis range: τ (reverse 1-0)
        ax.set_ylim(1, 0)  # Y-axis range: PF (reverse 1-0)
        ax.set_zlim(0, 1)  # Z-axis range: PD (0-1)

        # Adjust the view angle to correct the orientation of the axes
        ax.view_init(elev=20, azim=-140)  # Set view angle

        # Add black coordinate lines for 12 edges of the 3D plot
        ax.w_xaxis.line.set_color("black")
        ax.w_yaxis.line.set_color("black")
        ax.w_zaxis.line.set_color("black")

        # Set the 12 edges of the 3D axes to be black
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.w_xaxis.line.set_color("black")
        ax.w_yaxis.line.set_color("black")
        ax.w_zaxis.line.set_color("black")

        # Add grid, legend, and title
        ax.grid(False)  # Disable grid
        plt.legend()
        plt.title("3D ROC Curve with White Background")

        plt.show()  # Display the 3D ROC curve

        # Save the results (AUC metrics data) in a .npz file for further analysis
        # 保存 AUC_PDPF (ROC Curve) 的数据
        np.savez('auc_pdpf_data.npz',
                 fpr=fpr, tpr=tpr,
                 auc_pdpf=auc_pdpf)

        # 保存 AUC_PFT (Threshold vs False Positive Rate) 的数据
        np.savez('auc_pft_data.npz',
                 thresholds=thresholds, fpr=fpr,
                 auc_pft=auc_pft)

        # 保存 AUC_PDT (Threshold vs True Positive Rate) 的数据
        np.savez('auc_pdt_data.npz',
                 thresholds=thresholds, tpr=tpr,
                 auc_pdt=auc_pdt)

        # 保存 3D ROC 曲线数据 (Threshold, False Positive Rate, True Positive Rate)
        np.savez('roc_3d_curve_data.npz',
                 thresholds=thresholds, fpr=fpr, tpr=tpr)

        # 打印保存完成的信息
        print("所有数据已分别保存为单独的 npz 文件：")
        print("1. auc_pdpf_data.npz")
        print("2. auc_pft_data.npz")
        print("3. auc_pdt_data.npz")
        print("4. roc_3d_curve_data.npz")
        # Visualize ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, marker='.')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        # Plot Separability Map
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
    plt.imshow(gt, cmap='viridis')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(detection_result, cmap='viridis')
    plt.title("Detection Result")
    plt.axis("off")
    plt.show()


def visualize_separability_boxplot(target_similarity, background_similarity):
    """
    Visualize separability using a boxplot where the target is on the left and background is on the right.
    Outliers are hidden for cleaner visualization.
    """
    plt.figure(figsize=(8, 6))

    # Reverse the order: target on the left, background on the right
    box = plt.boxplot([target_similarity, background_similarity],
                      labels=["Target", "Background"],
                      patch_artist=True,  # Enable filling colors
                      showmeans=True,  # Show means
                      meanline=True,  # Use lines to display the mean
                      showfliers=False)  # Hide outliers

    # Set box colors
    colors = ['red', 'green']  # Target is red, background is green
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set mean line colors
    for meanline, color in zip(box['means'], colors):
        meanline.set_color('black')  # Mean lines are black

    # Customize plot appearance
    plt.title("Separability Analysis: Boxplot of Similarity Metrics", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.ylim(0, 1)  # Set similarity range from 0 to 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines for reference

    # Show the plot
    plt.show()

def save_detection_result_image(detection_result, modelConfig, epoch):
    """
    将检测结果保存为图片文件。
    """
    # 创建保存路径
    save_path = os.path.join(modelConfig["save_dir"], 'detection_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 生成保存文件名
    filename = f"detection_result_epoch_{epoch}.png"
    filepath = os.path.join(save_path, filename)

    # 绘制检测结果
    plt.figure(figsize=(8, 6))
    plt.imshow(detection_result, cmap='viridis')
    # 删除标题行，不显示标题
    plt.axis('off')

    # 保存图片，去除多余的空白
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Detection result saved at: {filepath}")

