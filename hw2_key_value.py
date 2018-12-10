# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:29:41 2018

@author: LDC13

使用key-value 方式，制作数据集，每次读一行数据，直接将读取的数据转换成 [id][film] = score
"""


import numpy as np
import pandas as pd
import time

print('Saving Train data.........................')

train_path = "netflix_train.txt"
train_file = open(train_path)
train_data = {}     # 存放key-value [id][film] = score
time_start=time.time()

for line in train_file.readlines():    # 读取固定行[1:2]
    line = line.strip().split(' ')     # 读取一行，按照空格分离  line 为读取的列表数据
    user_ID = int(line[0])
    film_ID = int(line[1])
    Score = int(line[2])
    if not user_ID in train_data.keys():   # 如果字典中没有key,则生成key
        train_data[user_ID] = {film_ID:Score}
    else:
        train_data[user_ID][film_ID] = Score   # [key_row][key_column] = values
        
train_data_df = pd.DataFrame(train_data).T.fillna(0)   # 转化成df
print('train_data_df:',train_data_df.shape)
train_data_df.to_csv('Xtrain.csv',header=True,index=True)      # 保存行和列名  
time_end=time.time()
print('Time Cost:',time_end-time_start,'s')



print('Saving Test data.........................')
test_path = "netflix_test.txt"
test_file = open(test_path)
test_data = {}     # 存放key-value [id][film] = score
time_start_1 = time.time()

for line in test_file.readlines():
    line = line.strip().split(' ')
    user_ID = int(line[0])
    film_ID = int(line[1])
    Score = int(line[2])
    if not user_ID in test_data.keys():
        test_data[user_ID] = {film_ID:Score}
    else:
        test_data[user_ID][film_ID] = Score
        
test_data_df = pd.DataFrame(test_data).T.fillna(0)
print('test_data_df:',test_data_df.shape)
test_data_df.to_csv('Xtest.csv',header=True,index=True)     
time_end_1 =time.time()
print('Time Cost:',time_end_1 - time_start_1,'s')



#Xtrain = pd.read_csv('Xtrain.csv')
#Xtest = pd.read_csv('Xtest.csv')
## Xtest 列不足，将其补足
#
#col = list(Xtest.columns)   # 获得Xtest的列名
#Col = col[1:]
#results = list(map(int,Col))  # 将列名转换成整数便于查询
#missing = []   # 保存缺失值
#for i in range(10000):
#    j = i + 1
#    if j not in results:
#        print (j)
#        missing.append(j)
#for i in range(len(missing)):
#    Xtest[str(missing[i])] = 0
#Xtest.to_csv('Xtest_2.csv',header=True,index=True) 
 



        

    
    


