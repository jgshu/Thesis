import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from .utils import find_files

def data_normalization(base_path, type_num, sum_flag=False):
    type_num_after_anomaly_detection_path =  base_path + 'data/type_%s_after_anomaly_detection/' % type_num
    type_num_normalization_path = base_path + 'data/type_%s_normalization/' % type_num
    sum_filename = 'type%s_%s' % (type_num, type_num)
    file_names_list = find_files(type_num_after_anomaly_detection_path)

    if sum_flag:
        print('Normalization of sum file...')
        file_names_list = [sum_filename]
    else:
        print('Normalization of single file...')
        file_names_list.remove(sum_filename)

    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        print('-------' + co_name + '--------')
        print('-------' + user_id + '--------')

        # 导入行业x的数据
        df = pd.read_csv(type_num_after_anomaly_detection_path + file_name + '.csv')

        # 输出csv标准化文件和npy最大最小值文件
        # print('Normalization...')
        # TODO: 提取需要归一化的特征（特征改变需要重新设置！！！）
        features = df.columns.values[-6:]

        scaler_df = df.loc[:, features]
        # 归一化
        scaler = MinMaxScaler()  # 实例化
        scaler = scaler.fit(scaler_df)  # fit，在这里本质是生成min(x)和max(x)
        result = scaler.transform(scaler_df)  # 通过接口导出结果
        for i in range(len(features)):
            feature = features[i]
            df[feature] = result[:, i]
        print('Saving %s - %s csv file...' % (co_name, user_id))

        if not os.path.exists(type_num_normalization_path):
            os.makedirs(type_num_normalization_path)
        df.to_csv(type_num_normalization_path + '%s_%s.csv' % (co_name, user_id), index=False)
