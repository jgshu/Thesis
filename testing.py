import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
import logging
from .data_preprocessing.utils import find_files
import eval


def denormalization(df, norm):
    # TODO: 提取需要反归一化的特征（特征改变需要重新设置！！！）
    features = df.columns.values[-1]
    scaler_df = df.loc[:, features]
    scaler = MinMaxScaler()
    if norm is 'standard':
        scaler = StandardScaler()
    scaler = scaler.fit(scaler_df)  # fit，生成min(x)和max(x)

    return scaler


def testing(base_path, args, specific_user_id):
    # 超参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 导入要inverse transform的df数据,生成userID和coName的键值对
    type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % args.type_num
    file_names_list = find_files(type_num_after_anomaly_detection_path)

    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        if user_id == specific_user_id:
            logging.info('-------' + co_name + '--------')
            logging.info('-------' + user_id + '--------')
            df = pd.read_csv(type_num_after_anomaly_detection_path + file_name + '.csv')
            break

    scaler = denormalization(df, args.norm)

    # 导入测试集
    normalization_tvt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tvt/' % (
        args.type_num, args.norm, args.day_range)

    test_x = np.load(normalization_tvt_path + file_name + 'test_x_range_%s.npy' % args.train_range)
    test_y = np.load(normalization_tvt_path + file_name + 'test_y_range_%s.npy' % args.train_range)
    test_x = torch.from_numpy(test_x).type(torch.Tensor)
    test_y = torch.from_numpy(test_y).type(torch.Tensor)

    model = torch.load(base_path + 'models/model.pkl')
    model.eval()

    # 模型用于测试集
    test_x = test_x.to(device)
    test_y_pred = model(test_x).to(device)

    test_y = scaler.inverse_transform(test_y)
    # print(test_y.shape)
    test_y_pred = scaler.inverse_transform(test_y_pred.detach().cpu().numpy())
    # print(test_y_pred.shape)

    # evaluation
    MAE = eval.calcMAE(test_y[:, 0], test_y_pred[:, 0])
    print("test MAE", MAE)
    MRSE = eval.calcRMSE(test_y[:, 0], test_y_pred[:, 0])
    print("test RMSE", MRSE)
    MAPE = eval.calcMAPE(test_y[:, 0], test_y_pred[:, 0])
    print("test MAPE", MAPE)
    SMAPE = eval.calcSMAPE(test_y[:, 0], test_y_pred[:, 0])
    print("test SMAPE", SMAPE)

