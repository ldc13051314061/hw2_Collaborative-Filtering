# hw2_Collaborative-Filtering
大数据分析作业2
1. 制作数据集
hw2_key_value.py； 使用key-value方式生成数据集
每次读一行数据，直接将读取的数据转换成 [id][film] = score
参考连接： https://blog.csdn.net/qq_25948717/article/details/81839463
采用python字典来表示每位用户评论的电影和评分

Xtrain.csv: df 10000*10000   训练集用户对电影的评分矩阵
Xtest.csv:df 10000*9963     测试集用户对电影的评分矩阵
Xtest_2.csv: df 10000*10000  测试集用户对电影的评分矩阵,用0补全缺失列,已经排序
COSINE_MATRIX_2.txt:mat 10000*10000   用户间的相似性矩阵    对角线元素为0，使用空格分开
Xtrain_data_mat.txt：mat 10000 * 10000   训练集评分矩阵 使用空格分开
Xtest_data_mat.txt：mat 10000 * 10000   测试集评分矩阵 使用空格分开
SCORE_predict.txt: mat 10000 * 10000   预测评分矩阵 使用空格分开

2. 预测结果与误差
X_test 添加列补零，排序
余弦相似度，利用矩阵计算提高计算速度
RMSE: 0.9354785973697769

3.FunkSVD算法用于推荐
迭代时注意保存U1,U2,V1,V2
使用F范数,计算RMSE,比较快
