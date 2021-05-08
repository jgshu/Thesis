import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

type_num = 3

print('Loading data...')
# 导入数据
csv_path = './data/type_%s_csv_after_anomaly_detection/' % type_num
file_list = os.listdir(csv_path)
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    print(coName)
    user_id_co_name_dict[userID] = coName
    df = pd.read_csv(csv_path + file)
    user_id_df_dict[userID] = df

print('Normalization...')
# 输出csv标准化文件和npy最大最小值文件
for userID in user_id_df_dict.keys():
    coName = user_id_co_name_dict[userID]
    user_df = user_id_df_dict[userID].copy()
    # 提取需要归一化的特征（特征改变需要重新设置！！！）
    features = user_df.columns.values[-6:]

    scaler_df = user_df.loc[:, features]
    # 归一化
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(scaler_df)  # fit，在这里本质是生成min(x)和max(x)
    result = scaler.transform(scaler_df)  # 通过接口导出结果
    for i in range(len(features)):
        feature = features[i]
        user_df[feature] = result[:, i]
    print('Saving %s - %s csv file...' % (coName, userID))
    path = './data/type_%s_csv_normalization/' % type_num
    if not os.path.exists(path):
        os.makedirs(path)
    user_df.to_csv(path + '%s_%s.csv' % (coName, userID), index=False)
