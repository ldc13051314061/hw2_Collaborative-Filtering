# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:56:56 2018

@author: LDC13
基于梯度的矩阵分解算法
FunkSVD
"""
import numpy as np
from numpy import random,mat 
import pandas as pd
import time
from math import *
np.seterr(invalid='ignore')

# 读取数据集
print('begin loading --------')
Xtrain_data = np.loadtxt("Xtrain_data_mat.txt",delimiter=' ')
Xtest_data = np.loadtxt("Xtest_data_mat.txt",delimiter=' ')
# k,lamda是参数
A = Xtrain_data.copy()
# 矩阵A在有评价值得地方则为1，没有评价值得地方则为0
A[A>0] = 1   # 矩阵A代表评分位置  
X = Xtrain_data.copy()

def FunkSVD(k,lamda,A,X,lr,steps):
    alpha = lr   # 学习率
    (m,n) = A.shape   # 行和列  
    U_1 = np.random.random((m,k))/100
    V_1 = np.random.random((n,k))/100
    U_2 = np.random.random((m,k))/100
    V_2 = np.random.random((n,k))/100
    for i in range(steps):
        U_1 = U_2
        V_1 = V_2
        J = 1/2 * (np.square(np.linalg.norm(A * (X - np.dot(U_1,V_1.T)),ord = 'fro'))) + lamda * np.square(np.linalg.norm(U_1,ord = 'fro')) + lamda * np.linalg.norm(V_1,ord = 'fro')
        print('steps:',i,'J=',J)
        if (J<0.01):
            print('converge')
            break
        U_2 = U_1 - alpha * (np.dot((A *(np.dot(U_1,V_1.T) - X)),V_1) + 2*lamda*U_1)
        V_2 = V_1 - alpha * (np.dot((A *(np.dot(U_1,V_1.T) - X)).T,U_1) + 2*lamda*V_1)
        
    return U_1,V_1

N = 5
M = 10

Results = np.zeros((N,M))
for i in range(N):
    k = 10*(i+1)
    t1 = time.time()
    for j in range(M):
        lamda = 0.0001 * (j+1)
        Pre_U,Pre_V = FunkSVD(k,lamda,A,X,lr=0.0001,steps= 50)
        predict = np.dot(Pre_U,Pre_V.T)
        error = Xtest_data - predict
        RMSE = 1/10000 * (np.linalg.norm(error,ord = 'fro'))
        print(i,j,'RMSE:',RMSE)
        Results[i,j] = RMSE
    t2 = time.time()
    cost = t2-t1
    print('Cost time:',cost,'s')
    
np.savetxt('Results1.txt', Results, delimiter=' ')
        
# 当lambda=0.0001 - 0.001,k=30,查看迭代图

N = 5
M = 10

Results = np.zeros((N,M))
for i in range(N):
    k = 1000*(i+1)
    t1 = time.time()
    for j in range(M):
        lamda = 0.0001 * (j+1)
        Pre_U,Pre_V = FunkSVD(k,lamda,A,X,lr=0.0001,steps= 10)
        predict = np.dot(Pre_U,Pre_V.T)
        error = Xtest_data - predict
        RMSE = 1/10000 * (np.linalg.norm(error,ord = 'fro'))
        print(i,j,'RMSE:',RMSE)
        Results[i,j] = RMSE
    t2 = time.time()
    cost = t2-t1
    print('Cost time:',cost,'s')
    
np.savetxt('Results1'+'k1000'+'lr0.0001'+'steps10'+'.txt', Results, delimiter=' ')
#





N = 5
M = 10

Results = np.zeros((N,M))
for i in range(N):
    k = 10*(i+1)
    t1 = time.time()
    for j in range(M):
        lamda = 0.0001 * (j+1)
        Pre_U,Pre_V = FunkSVD(k,lamda,A,X,lr=0.0001,steps= 100)
        predict = np.dot(Pre_U,Pre_V.T)
        error = Xtest_data - predict
        RMSE = 1/10000 * (np.linalg.norm(error,ord = 'fro'))
        print(i,j,'RMSE:',RMSE)
        Results[i,j] = RMSE
    t2 = time.time()
    cost = t2-t1
    print('Cost time:',cost,'s')
    
np.savetxt('Results2'+'lr0.0001'+'steps100'+'.txt', Results, delimiter=' ')

N = 5
M = 10

Results = np.zeros((N,M))
for i in range(N):
    k = 10*(i+1)
    t1 = time.time()
    for j in range(M):
        lamda = 0.0001 * (j+1)
        Pre_U,Pre_V = FunkSVD(k,lamda,A,X,lr=0.001,steps= 50)
        predict = np.dot(Pre_U,Pre_V.T)
        error = Xtest_data - predict
        RMSE = 1/10000 * (np.linalg.norm(error,ord = 'fro'))
        print(i,j,'RMSE:',RMSE)
        Results[i,j] = RMSE
    t2 = time.time()
    cost = t2-t1
    print('Cost time:',cost,'s')
    
np.savetxt('Results3'+'lr0.001'+'steps50'+'.txt', Results, delimiter=' ')         
        
    
    
            
        
    
    
    

