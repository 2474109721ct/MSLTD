import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    min_x = np.min(X)
    max_x = np.max(X)
    if min_x == max_x:
        return np.zeros_like(X)
    return np.float32((X - min_x) / (max_x - min_x))


def sofzt(X, lamd):
    '''
    Soft thresholding function
    :param X: input
    :param lamd: threshold
    :return:
    '''
    t1 = np.sign(X)
    t2 = np.abs(X) - lamd
    t2[t2 < 0] = 0
    return t1 * t2


def SAMDetector(X, P):
    '''
    SAM for target detection, the smaller the angle is, the more similar the two vectors are
    :param X: The image X in matrix form L*N
    :param P: The target prior P with a size of L*1
    :return:
    '''
    norm_X = np.sqrt(np.sum(np.square(X), axis=0))
    norm_P = np.sqrt(np.sum(np.square(P), axis=0))
    print(f"newX: {X.shape}")
    print(f"NEWP: {P.shape}")
    x_dot_P = np.sum(X * P, axis=0)

    value = x_dot_P / (norm_X * norm_P)

    angle = np.arccos(np.clip(value, -1, 1))
    return angle * 180 / np.pi


def readData(data_no):
    '''
    read data
    :param data_no: 0 means abu2, 1 means sandiego2
    :return:
    '''
    if data_no == 0:
        path = 'abu2.mat'
        Kt = 20
    elif data_no == 1:
        path = 'abu1.mat'
        Kt = 20
    else:
        print('invalid data')
        return

    mat = sio.loadmat(path)
    hs = mat['data']
    groundtruth = mat['map']

    hs = standard(hs)
    H, W, L = hs.shape
    print(f"重塑后的数据形状: {hs.shape}")
    print(f"重塑后的地面真值形状: {groundtruth.shape}")
    hs_matrix = np.reshape(hs, [-1, L], order='F')
    hs_matrix = hs_matrix.T  ## L * N
    print(f"重塑后的数据形状: {hs_matrix.shape}")
    print(f"重塑后的地面真值形状: {groundtruth.shape}")
    target_prior = mat['d']
    target_prior = standard(target_prior)

    return hs_matrix, groundtruth, target_prior, H, W, L, Kt


def dictionaryConstruction(X, p, Kt, m=25, n=5):
    # 打印 X 和 p 的大小
    print(f"X shape: {X.shape}")
    print(f"p shape: {p.shape}")

    L, N = X.shape
    detection_result = SAMDetector(X, p)
    print(f"detection_result: {detection_result}")
    print(f"Unique values in detection_result: {np.unique(detection_result).size}")
    print(f"SAMX shape: {X.shape}")
    print(f"SAMp shape: {p.shape}")
    ind = np.argsort(detection_result)
    ind = ind.flatten(order='F')
    ind = ind[:Kt]
    At = X.T[ind]
    print(f"AT: {At.shape}")
    target_map = np.zeros([N])
    print(f"target_map: {target_map.shape}")
    target_map[ind] = 1
    print(f"target_map1: {target_map.shape}")
    X = X.T[target_map == 0]
    estimator = KMeans(n_clusters=m)
    print(f"fit X: {X.shape}")
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

    return Ab.T, At.T


if __name__ == '__main__':
    start = time.perf_counter()
    ## read data, 0 means abu2, and 1 means sandiego2
    X, gt, p, H, W, L, Kt = readData(1)
    ## construct the background dictionary and target dictionary
    data = X
    # 打印 X 和 p 的大小
    print(f"X shape: {X.shape}")
    print(f"p shape: {p.shape}")

    Ab, At = dictionaryConstruction(X, p, Kt=Kt, m=25, n=5)
    sio.savemat('./abu1/back.mat', {'back': Ab})
    # TargetTrain = p
    # sio.savemat('original_target.mat', {'TargetTrain': TargetTrain})
    # sio.savemat('target.mat', {'target': At})
