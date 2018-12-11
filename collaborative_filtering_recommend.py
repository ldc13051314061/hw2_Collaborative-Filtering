# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:46:52 2018
使用协同过滤算法进行推荐
@author: LDC13
"""
import numpy as np
from numpy import random,mat
import pandas as pd
import time
from math import *
#
Xtrain = pd.read_csv('Xtrain.csv')
Xtest = pd.read_csv('Xtest_2.csv')

# Xtest 列不足，将其补足
col = list(Xtest.columns)   # 获得Xtest的列名
Col = col[1:]
results = list(map(int,Col))  # 将列名转换成整数便于查询
missing = []   # 保存缺失值
for i in range(9999):
    j = i+1 
    if j not in results:
        print (j)
        missing.append(j)
for i in range(len(missing)):
    Xtest[str(missing[i])] = 0
    
# 对列排序
list1 = ['Unnamed: 0']
list2 = []
for i in range(10000):
    list2.append(i+1)
list3 = list(map(str,list2))
list_order = list1 + list3
Xtest = Xtest[list_order]
Xtest.to_csv('Xtest_2.csv',header=True,index=True)    

'''
Collaborative Filtering Algorithom
'''
Xtrain_tmp = Xtrain.values   # 第一列为行名
Xtest_tmp = Xtrain.values 
Xtrain_data = Xtrain_tmp[:,1:]  # 去掉第一列
Xtest_data = Xtest_tmp[:,1:]    # 行号代表user  
Xtrain_data_mat = mat(Xtrain_data)   # ndarrray 转换成 matrix
Xtest_data_mat = mat(Xtest_data)
#np.savetxt('Xtrain_data_mat.txt', Xtrain_data_mat, delimiter=' ')   # 保存
#np.savetxt('Xtest_data_mat.txt', Xtest_data_mat, delimiter=' ')   # 保存


# 两个矩阵计算相似度，按行计算相似度
# 两矩阵计算相似度应为同维度
# 返回值RES为A矩阵每行对B矩阵每行的向量余弦值
# RES[i,j] 表示A矩阵第i行向量和B矩阵第j行向量余弦相似度


#Xtrain_data_mat = np.loadtxt("Xtrain_data_mat.txt",delimiter=' ')
#Xtest_data_mat = np.loadtxt("Xtest_data_mat.txt",delimiter=' ')
start = time.time()
#
#start = time.time()
#
def cosine_Matrix(_matrixA,_matrixB):
    # 乘以转置 点积
    _matrixA_matrixB = _matrixA * _matrixB.transpose()
    # 按行求和，生成一个列向量，即各行向量的模
    
    _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum(axis=1))
    return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

# 向量计算余弦相似度
#def cosine(_vec1,_vec2):
#    return float(np.sum(_vec1 * _vec2))/(np.linalg.norm(_vec1) * np.linalg.norm(_vec2))

COSINE_MATRIX = np.zeros((10000,10000))
COSINE_MATRIX = cosine_Matrix(Xtrain_data_mat,Xtrain_data_mat)

#np.savetxt('COSINE_MATRIX.txt', COSINE_MATRIX, delimiter=' ')   # 保存

# 对角元素变为0
for i in range(10000):
    COSINE_MATRIX[i,i] = 0
#np.savetxt('COSINE_MATRIX_2.txt', COSINE_MATRIX, delimiter=' ')
#
##  读取相似度矩阵txt文件
#
#start = time.time()
#COSINE_MATRIX = np.loadtxt("COSINE_MATRIX_2.txt",delimiter=' ')
#Numerator = COSINE_MATRIX * Xtrain_data_mat.T    # 分子
Numerator  = np.dot(COSINE_MATRIX , Xtrain_data_mat.T)
COSINE_MATRIX[Xtrain_data_mat == 0] = 0   # 用户对电影未评分，相似度应为0，权重置为0
Denominator = COSINE_MATRIX.sum(axis=1)   # 分母，是一个一维向量

SCORE_predict = np.divide(Numerator.T,Denominator)
SCORE_predict = SCORE_predict.T    # 预测得分
#np.savetxt('SCORE_predict_1.txt', SCORE_predict, delimiter=' ')   # 保存

# 计算rmse
SCORE_predict[Xtest_data_mat == 0] = 0 #  测试集中为0的数据，不进行预测相减
error = SCORE_predict - Xtest_data_mat   # 求差值
RMSE = 1/10000 * (np.linalg.norm(error,ord='fro'))
end = time.time()
print('RMSE:',RMSE)
print('Time cost:',end-start)






    
    
    
    
    

    
    
    
    
    
    
    



