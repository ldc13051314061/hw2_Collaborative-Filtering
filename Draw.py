# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:21:11 2018
画图
@author: LDC13
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

filrname = "Results1k1000lr0.0001steps10.txt"
Result = np.loadtxt(filrname,delimiter=' ')
(n,m) = Result.shape
sub_axix = []
lamda = 0.0001
for i in range(m):
    sub_axix.append(lamda * (i+1))
plt.title('RMSE (lr=0.0001 steps= 10)')
plt.plot(sub_axix,Result[0,:],'*-',color='green',label='k=1000')
plt.plot(sub_axix,Result[1,:],'*-',color='red',label='k=2000')
plt.plot(sub_axix,Result[2,:],'*-',color='skyblue',label='k=3000')
plt.plot(sub_axix,Result[3,:],'*-',color='blue',label='k=4000')
plt.plot(sub_axix,Result[4,:],'*-',color='yellow',label='k=5000')
plt.legend()

plt.xlabel('$\lambda$')
plt.ylabel('RMSE')
plt.show()

    


