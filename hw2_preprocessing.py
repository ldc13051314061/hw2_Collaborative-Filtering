# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:25:24 2018
hw2 数据读取
@author: LDC13
"""

import numpy as np
import pandas as pd


# 导入数据
# 用户列表users.txt： 10000行，每行一个整数，表示用户的id，文件对应本次作业的所有用户
user_path = "users.txt"
user_data = np.loadtxt(user_path)    # (10000,)
user_data_df = pd.DataFrame(user_data, columns=['ID'])
print('====user=====')
print(user_data_df.head())

# 训练集netflix_train.txt：包含689万条用户打分，每行为一次打分，包括用户id、电影id、分数和打分日期，
# 其中用户id均出现在users.txt中，电影id为1-10000的整数
train_path = "netflix_train.txt"
train_data_df = pd.read_table(train_path,sep=' ',names=['ID', 'FilmID','Score','Date'])   #  (6897745, 4)
print('====train====')
print(train_data_df.head())

# 测试集netflix_test.txt：包含172万条用户打分，格式同上
test_path = "netflix_test.txt"
test_data_df = pd.read_table(test_path,sep=' ',names=['ID', 'FilmID','Score','Date'])    #  (1719465, 4)
print('===test====')
print(test_data_df.head())



        

    
    


