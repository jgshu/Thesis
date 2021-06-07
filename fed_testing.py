import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import os
import logging

from data_preprocessing.utils import find_files
from test_plotting import test_plotting

from eval import *

def denormalization(df, norm):
    # TODO: 提取需要反归一化的特征（特征改变需要重新设置！！！）
    # features = df.columns.values[-1]
    scaler_df = df.loc[:, ['load']]
    scaler = MinMaxScaler()
    if norm is 'standard':
        scaler = StandardScaler()
    scaler = scaler.fit(scaler_df)  # fit，生成min(x)和max(x)

    return scaler


def eval(test_y, test_y_pred):
    # evaluation
    MAE = calcMAE(test_y, test_y_pred)
    print("test MAE", MAE)
    MRSE = calcRMSE(test_y, test_y_pred)
    print("test RMSE", MRSE)
    MAPE = calcMAPE(test_y, test_y_pred)
    print("test MAPE", MAPE)
    SMAPE = calcSMAPE(test_y, test_y_pred)
    print("test SMAPE", SMAPE)


# 逐点预测point-to-point prediction
def point_to_point(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id, device):
    test_y = scaler.inverse_transform(test_y[:, 26])
    for i in range(int(args.epochs / args.frequency_of_the_test)):
        try:
            file_name = 'model_%02d' % i
            print('-------%s-------' % file_name)
            model.load_state_dict(torch.load(model_path + file_name + '.pth'))
            model.eval()
            test_x_tensor = torch.from_numpy(test_x).type(torch.Tensor)
            test_x_tensor = test_x_tensor.to(device)
            test_y_pred_tensor = model(test_x_tensor).to(device)
            test_y_pred = scaler.inverse_transform(test_y_pred_tensor.detach().cpu().numpy()[:, 26])

            eval(test_y, test_y_pred)
            test_plotting(test_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id)

            del test_x_tensor
            del test_y_pred_tensor
            del test_y_pred

        except Exception as e:
            print(e)

# 全序列预测 full sequence prediction
def full_sequence(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id, device):
    test_y = scaler.inverse_transform(test_y[:args.seq_len, 26])
    for i in range(int(args.epochs / args.frequency_of_the_test)):
        try:
            file_name = 'model_%02d' % i
            print('-------%s-------' % file_name)
            model.load_state_dict(torch.load(model_path + file_name + '.pth'))
            model.eval()
            pred_list = []
            batch_test_x = test_x[:1, :, :]
            for j in range(args.seq_len):
                batch_test_x_tensor = torch.from_numpy(batch_test_x).type(torch.Tensor)
                batch_test_x_tensor = batch_test_x_tensor.to(device)
                batch_test_y_pred_tensor = model(batch_test_x_tensor).to(device)
                pred_list.append(batch_test_y_pred_tensor.detach().cpu().numpy()[0, :].tolist())
                batch_test_x = np.append(batch_test_x[:, 1:, :], [[pred_list[j]]], axis=1)

            pred_array = np.array([pred_list])
            test_y_pred = scaler.inverse_transform(pred_array[:, 26])

            eval(test_y, test_y_pred)
            test_plotting(test_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id)

            del batch_test_x
            del batch_test_x_tensor
            del batch_test_y_pred_tensor
            del test_y_pred

        except Exception as e:
            print(e)


def multi_sequence(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id, device):
    test_y = scaler.inverse_transform(test_y[:, 26])
    for i in range(int(args.epochs / args.frequency_of_the_test)):
        try:
            file_name = 'model_%02d' % i
            print('-------%s-------' % file_name)
            model.load_state_dict(torch.load(model_path + file_name + '.pth'))
            model.eval()
            pred_seq_list = []
            horizon = args.day_range
            for j in range(int(test_y.shape[0] / horizon)):
                pred_list = []
                batch_test_x = test_x[j * horizon: j * horizon + 1, :, :]
                for k in range(horizon):
                    batch_test_x_tensor = torch.from_numpy(batch_test_x).type(torch.Tensor)
                    batch_test_x_tensor = batch_test_x_tensor.to(device)
                    batch_test_y_pred_tensor = model(batch_test_x_tensor).to(device)
                    pred_list.append(batch_test_y_pred_tensor.detach().cpu().numpy()[0, :].tolist())
                    batch_test_x = np.append(batch_test_x[:, 1:, :], [[pred_list[k]]], axis=1)
                pred_seq_list.extend([out[26] for out in pred_list])

            pred_seq_array = np.array(pred_seq_list)
            test_y_pred = scaler.inverse_transform(pred_seq_array)
            test_y = test_y[:int(test_y.shape[0] / horizon) * horizon]

            eval(test_y, test_y_pred)
            test_plotting(test_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id)

            del batch_test_x
            del batch_test_x_tensor
            del batch_test_y_pred_tensor
            del test_y_pred

        except Exception as e:
            print(e)


def testing(base_path, model_path, args, dt_string, model, specific_user_id, type='p2p'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normalization_tvt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tvt/' % (
        args.type_num, args.day_range, args.norm)

    # 导入要inverse transform的df数据,生成userID和coName的键值对
    type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % args.type_num
    file_names_list = find_files(type_num_after_anomaly_detection_path)

    df = pd.DataFrame()
    specific_user_id_file_name = ''
    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        if user_id == specific_user_id:
            logging.info('-------' + co_name + '--------')
            logging.info('-------' + user_id + '--------')
            df = pd.read_csv(type_num_after_anomaly_detection_path + file_name + '.csv')
            specific_user_id_file_name = file_name
            break

    scaler = denormalization(df, args.norm)

    test_x = np.load(normalization_tvt_path + specific_user_id_file_name + '/test_x_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))
    test_y = np.load(normalization_tvt_path + specific_user_id_file_name + '/test_y_in_%s_out_%s_range_%s.npy' % (args.seq_len, args.out_features, args.train_range))

    if type == 'p2p':
        test_path = base_path +'output/type_%s/day_%s_range_%s_%s/model/%s/img_p2p/' \
                     % (args.type_num, args.day_range, args.train_range, args.norm, dt_string)
        point_to_point(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id,
                           device)
    elif type == 'fn':
        test_path = base_path +'output/type_%s/day_%s_range_%s_%s/model/%s/img_fs/' \
                     % (args.type_num, args.day_range, args.train_range, args.norm, dt_string)
        full_sequence(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id, device)
    elif type == 'ms':
        test_path = base_path +'output/type_%s/day_%s_range_%s_%s/model/%s/img_ms/' \
                     % (args.type_num, args.day_range, args.train_range, args.norm, dt_string)
        multi_sequence(test_path, model_path, args, test_x, test_y, scaler, dt_string, model, specific_user_id,
                           device)







