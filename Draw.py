# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:21:11 2018
画图
@author: LDC13
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

filrname = "Results_k20_lambda_0.01.txt"
Result = np.loadtxt(filrname,delimiter=' ')
(n,m) = Result.shape
x = np.arange(m)
#x2 = np.arange(100)
Min_RMSE = min(Result[1,:])

fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax1.set_title('$\lambda$ =0.01 k = 20')
ax1.plot(x,Result[0,:],label='$J$')
ax1.legend()
ax1.set_xlabel('steps')

ax2 = fig1.add_subplot(212)
ax2.plot(x,Result[1,:],label='$RMSE$')
#ax2.set_title('$\lambda$ =0.01 $\k$ = 50' )
ax2.legend()
ax2.set_xlabel('steps \n' + ' $RMS{E_{\min }}$ =' + str(Min_RMSE))
plt.show()




#sub_axix = []
#lamda = 0.0001
#for i in range(m):
#    sub_axix.append(lamda * (i+1))
#plt.title('RMSE (lr=0.0001 steps= 10)')
#plt.plot(sub_axix,Result[0,:],'*-',color='green',label='k=1000')
#plt.plot(sub_axix,Result[1,:],'*-',color='red',label='k=2000')
#plt.plot(sub_axix,Result[2,:],'*-',color='skyblue',label='k=3000')
#plt.plot(sub_axix,Result[3,:],'*-',color='blue',label='k=4000')
#plt.plot(sub_axix,Result[4,:],'*-',color='yellow',label='k=5000')
#plt.legend()
#
#plt.xlabel('$\lambda$')
#plt.ylabel('RMSE')
#plt.show()

    


