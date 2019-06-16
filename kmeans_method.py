# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:11:13 2019

@author: Xiang
"""

import numpy as np
import pandas as pd

# 計算歐式距離
def dist(vectorA, vectorB):
    return np.sqrt(np.sum(np.power(vectorA - vectorB, 2)))
def unif(number):
    return np.random.rand(number) #U~(0, 1)
def farthest_prob_points(datasets, k):
    #定義一個刪除資料的function 放進的data要是array
    def DeleteData(datasets_array, index):
        return np.delete(datasets_array, index, axis=0)
    datasets = pd.DataFrame(datasets)
    data_length, n = datasets.shape #obs跟變數個數
    init_pts = np.zeros((k, n)) #存放初始質心array
    first_init_centroid = datasets.sample(1 ,axis=0)
    init_pts[0] = np.array(first_init_centroid)
    #先隨機挑選一個資料當成第一個質心，之後從資料裡刪除，避免日後重複計算
    #而之後的質心也是一樣，只要找到一個，就從資料裡刪除
    data_array = DeleteData(datasets.values, first_init_centroid.index)
    for i in range(1, k):
        null = []
        for item in data_array:
            pts_dist = []
            for pts in init_pts:
                if np.count_nonzero(pts) != 0: #一開始init_pts為k x n的0矩陣，用此判斷只跟已經存入的質心做距離
                    pts_dist.append(dist(item, pts)) 
            null.append(np.power(np.min(pts_dist), 2)) #與最近的一個質心的SSE
        cdf = np.cumsum(null/sum(null)) #機率:(D(x))^2 \ sum D(x)^2，累加機率 = 1
        uniform = unif(1) #U~(0, 1)
        index = np.searchsorted(cdf, uniform) #找U~(0, 1)的值落在哪個區間
        init_pts[i] = data_array[index]
        data_array = DeleteData(data_array, index)
    return init_pts

def kmc2(data_array, k, chain_length=200, random_state=None):
    #設定權重與亂數種子
    obs, var = data_array.shape
    weights = np.ones(obs)/obs
    random_state = np.random.RandomState(None)
    # 存放質心的array
    centers = np.zeros((k, var))
    # 通過給定的一維數組數據產生隨機採樣第一個質心 and compute proposal
    rel_row = data_array[random_state.choice(obs, p = weights), :]
    centers[0] = rel_row
    pts_dist = []
    for item in data_array:
        pts_dist.append(round(np.power(dist(item, centers[0]), 2), 3)) 
    cdf = 0.5*(pts_dist/sum(pts_dist)) + 1/(2 * obs)# q(x)
    
    
    for i in range(k-1):
        # 隨機採樣 chain_length個落在 1~X.shape[0]的index，而每個被選取的機率為cdf
        random_index = random_state.choice(obs, size=(chain_length), p=cdf)
        # 再換成相對應的機率值
        random_value = cdf[random_index]
        # 計算兩兩距離
        pts_dist = []
        for item in data_array[random_index, :]:
            pts_dist.append(round(np.power(dist(item, centers[0:(i+1), :]), 2), 3))
        
        # 計算最近距離
        min_dist = np.min(np.array([pts_dist]).T, axis=1)
        
        # 計算相對應的機率值(類似kmean++)
        curr_prob = response_p = random_state.random_sample(size=(chain_length))
        # Markov chain
        for j in range(chain_length):
            ratio = min_dist[j]/random_value[j]
            if j == 0 or curr_prob == 0.0 or ratio/curr_prob > response_p[j]:
                # 初始化            Metropolis-Hastings step
                curr_ind = j
                curr_prob = ratio
        rel_row = data_array[random_index[curr_ind], :]
        centers[i+1, :] = rel_row
    return centers