# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:01:41 2018

@author: LDC13
按照提交要求进行计算
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
    
    Results = np.zeros((2,steps))    # 保存每一步的函数值和测试集上的RMSE
    for i in range(steps):
        U_1 = U_2
        V_1 = V_2
        J = 1/2 * (np.square(np.linalg.norm(A * (X - np.dot(U_1,V_1.T)),ord = 'fro'))) + lamda * np.square(np.linalg.norm(U_1,ord = 'fro')) + lamda * np.linalg.norm(V_1,ord = 'fro')
        Results[0,i] = J
#        print('steps:',i,'J=',J)
        if (J<0.01):
            print('converge')
            break
        U_2 = U_1 - alpha * (np.dot((A *(np.dot(U_1,V_1.T) - X)),V_1) + 2*lamda*U_1)
        V_2 = V_1 - alpha * (np.dot((A *(np.dot(U_1,V_1.T) - X)).T,U_1) + 2*lamda*V_1)
        predict = np.dot(U_1,V_1.T)
        predict[Xtest_data<1]=0
        error = Xtest_data - predict
        RMSE = 1/10000 * (np.linalg.norm(error,ord = 'fro'))
        Results[1,i] = RMSE
        print('steps:',i,Results[0,i],Results[1,i])
    return Results

start = time.time()
Results = FunkSVD(50,0.01,A,X,lr=0.0001,steps=1000)
end = time.time()
np.savetxt('Results_k50_lambda_0.01.txt', Results, delimiter=' ')
print('Cost time:',end-start)


start = time.time()
Results = FunkSVD(20,0.001,A,X,lr=0.0001,steps=500)
end = time.time()
np.savetxt('Results_k20_lambda_0.001.txt', Results, delimiter=' ')
print('Cost time:',end-start)

start = time.time()
Results = FunkSVD(20,0.1,A,X,lr=0.0001,steps=500)
end = time.time()
np.savetxt('Results_k20_lambda_0.1.txt', Results, delimiter=' ')
print('Cost time:',end-start)

start = time.time()
Results = FunkSVD(20,0.0001,A,X,lr=0.0001,steps=500)
end = time.time()
np.savetxt('Results_k20_lambda_0.0001.txt', Results, delimiter=' ')
print('Cost time:',end-start)

start = time.time()
Results = FunkSVD(50,0.1,A,X,lr=0.0001,steps=500)
end = time.time()
np.savetxt('Results_k50_lambda_0.1.txt', Results, delimiter=' ')
print('Cost time:',end-start)

start = time.time()
Results = FunkSVD(50,0.001,A,X,lr=0.0001,steps=500)
end = time.time()
np.savetxt('Results_k50_lambda_0.001.txt', Results, delimiter=' ')
print('Cost time:',end-start)
