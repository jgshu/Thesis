import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from data_preprocessing.split_data_by_trade_type import split_data_by_trade_type
from data_preprocessing.feature_engineering import feature_engineering
from data_preprocessing.anomaly_detection import anomaly_detection
from data_preprocessing.data_normalization import data_normalization
from data_preprocessing.train_test_split import train_test_split


def data_preprocessing(base_path, type_num):
    # split_data_by_trade_type(base_path)
    # feature_engineering(base_path, type_num)
    # sum_user_id_list = ['638024734', '662615482']
    # feature_engineering(base_path, type_num, sum_flag=True, sum_user_id_list=sum_user_id_list)
    # anomaly_detection(base_path, type_num)
    # anomaly_detection(base_path, type_num, sum_flag=True)
    # data_normalization(base_path, type_num)
    # data_normalization(base_path, type_num, sum_flag=True)
    train_test_split(base_path, type_num, 96 * 7, 96, 365, 31, sum_flag=False)
    train_test_split(base_path, type_num, 96 * 7, 96, 365, 31, sum_flag=True)


if __name__ == '__main__':
    base_path = '../../../Downloads/Thesis-temp/'
    type_num = 10
    data_preprocessing(base_path, type_num)