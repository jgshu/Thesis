import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from .utils import find_files


def normalization(df, norm):
    # TODO: 提取需要归一化的特征（特征改变需要重新设置！！！）
    features = df.columns.values[-6:]
    scaler_df = df.loc[:, features]
    scaler = MinMaxScaler()
    if norm is 'standard':
        scaler = StandardScaler()
    scaler = scaler.fit(scaler_df)  # fit，生成min(x)和max(x)
    result = scaler.transform(scaler_df)  # 通过接口导出结果
    for i in range(len(features)):
        feature = features[i]
        df[feature] = result[:, i]
    return df


def data_normalization(base_path, type_num, day_range=96, norm='minmax', sum_flag=False):
    type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % type_num
    type_num_normalization_path = base_path + 'data/type_%s/day_%s/%s_normalization/' % (type_num, day_range, norm)

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

        # 归一化
        df = normalization(df, norm)

        # 切片
        slide_range = int(96 / day_range)
        df = df[::slide_range]

        # 输出csv文件
        print('Saving %s - %s csv file...' % (co_name, user_id))
        if not os.path.exists(type_num_normalization_path):
            os.makedirs(type_num_normalization_path)
        df.to_csv(type_num_normalization_path + '%s_%s.csv' % (co_name, user_id), index=False)
