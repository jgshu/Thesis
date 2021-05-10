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
from daily_load_plotting import daily_load_plotting


def before_normalization(base_path, type_num):
    # split_data_by_trade_type(base_path)
    feature_engineering(base_path, type_num)
    sum_user_id_list = ['638024734', '662615482']
    feature_engineering(base_path, type_num, sum_flag=True, sum_user_id_list=sum_user_id_list)
    anomaly_detection(base_path, type_num)
    anomaly_detection(base_path, type_num, sum_flag=True)

def normalization_and_split(base_path, type_num, day_range=96, norm='minmax'):
    data_normalization(base_path, type_num, day_range=day_range, norm=norm)
    data_normalization(base_path, type_num, day_range=day_range, norm=norm, sum_flag=True)
    # if day_range == 96:
    #     n_predictions = 96 * 7
    #     n_next = 96
    # elif day_range == 24:
    #     n_predictions = 24 * 7
    #     n_next = 24
    # train_test_split(base_path, type_num, n_predictions, n_next, 365, 31, day_range=day_range, norm=norm, sum_flag=False)
    # train_test_split(base_path, type_num, n_predictions, n_next, 365, 31, day_range=day_range, norm=norm, sum_flag=True)


def data_preprocessing(base_path, type_num):
    # before_normalization(base_path, type_num)
    # normalization_and_split(base_path, type_num, day_range=24, norm='minmax')
    # normalization_and_split(base_path, type_num, day_range=24, norm='standard')
    normalization_and_split(base_path, type_num, day_range=48, norm='minmax')
    normalization_and_split(base_path, type_num, day_range=48, norm='standard')
    # normalization_and_split(base_path, type_num, day_range=96, norm='minmax')
    # normalization_and_split(base_path, type_num, day_range=96, norm='standard')

if __name__ == '__main__':
    base_path = '../../../Downloads/Thesis-temp/'
    type_num = 10
    data_preprocessing(base_path, type_num)
    # daily_load_plotting(base_path, type_num, day_range=96, norm='minmax')
    daily_load_plotting(base_path, type_num, day_range=48, norm='minmax')
