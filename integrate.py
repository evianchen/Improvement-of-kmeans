# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:05:47 2019

@author: Xiang
"""

import numpy as np
import pandas as pd
import kmeans_method as km
def kmeans(datasets, k, convergence = 1e-6, method = True):
    """
    #參數說明#
    datasets: 資料集
    k: 分群個數
    convergence: 收斂條件，預設為10^-6
    method: True為AFKMC2，False為kmeans++
    
    #返回#
    begining_centroids: 初始質心位置
    cluster_centroids(k, var): 最終質心位置
    null_clusterAssign: 標籤
    np.sum(SSE_cluster): 各標籤到各自質心的SSE總和
    times: 迭代次數
    
    #使用#
    kmeans(X, 3, convergence=1e-6, method=True)
    """
    copydata = datasets.copy() #給AFKMC2重製用
    datasets = pd.DataFrame(datasets) 
    obs, var = datasets.shape #obs跟變數個數 
    #計算每次迭代後的質心
    #分別計算每一群中各個變數的mean，但並不含clusterAssign這一行
    def cluster_centroids(k, var):
        cluster_centroids = np.zeros((k, var))
        for i in range(k):
            cluster_centroids[i] = np.mean(datasets[datasets.iloc[:, var] == i+1])[0:var]
        return cluster_centroids
    
    #初始化質心
    if method == True:
        begining_centroids = init_cluster_centroids = km.kmc2(datasets.values, k)
    elif method == False:
        begining_centroids = init_cluster_centroids = km.farthest_prob_points(datasets, k)
    
    #初始化分群
    clusterAssign = np.repeat(None, obs)#先生成空值，之後再賦群值
    #合併資料跟clusterAssign
    datasets = pd.concat([datasets, pd.Series(clusterAssign, index = datasets.index)], 
                          axis = 1, ignore_index = True)
    
    #print("The begining centroids: \n", begining_centroids)
    #設定初始迭代次數為0，迭代到設定的次數，或者收斂就停止
    times = 1
    while True:       
        #計算每一筆obs跟k個質心的距離，並返回最小距離的index 代表更新到第幾群
        #k個為一組進行比較(因每個obs都要跟k個質心做距離計算)
        null_clusterAssign = [] #空列表，存入更新後的索引
        SSE_cluster = [] #空列表，計算SSE
        for item in np.array(datasets.drop(var, axis=1)):
            null_dist = np.zeros((k, 1))
            for j in range(k):
                null_dist[j] =(np.array(km.dist(item, init_cluster_centroids[j])))
            #返回最小值所在的索引，因k=1,2,..  因此返回後+1
            null_clusterAssign.append(null_dist.argmin() + 1)
            SSE_cluster.append(min(null_dist) ** 2) #歐式距離平方 = SSE
           
        #若reassign時，有某一個k沒被分配到，則重新找一開始的質心
        if len(np.unique(null_clusterAssign)) != len(range(k)):
            if method == True:
                init_cluster_centroids = km.kmc2(copydata, k)
            elif method == False:
                init_cluster_centroids = km.farthest_prob_points(datasets, k)
            continue
        
        #更新群
        datasets.iloc[:, var] = np.array(null_clusterAssign)
       
        #若前一次的質心位置跟重新assign的質心位置的距離差 < 收斂值，則結束迴圈
        #否則就將新assign的質心位置定成初始質心
        if km.dist(cluster_centroids(k, var), init_cluster_centroids) < convergence:
            break
        
        init_cluster_centroids = cluster_centroids(k, var) #設定初始質心，並持續更新
        times = times + 1
    return begining_centroids, cluster_centroids(k, var), null_clusterAssign, np.sum(SSE_cluster), times