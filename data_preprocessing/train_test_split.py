import pandas as pd
import numpy as np
import os
from .utils import find_files
from .utils import create_dataset

def train_test_split(base_path, type_num, n_predictions, n_next, train_range, test_range, sum_flag=False):
    # 可调参数
    date_range = 90
    n_predictions = 96 * 7
    n_next = 96
    # n_predictions = 96
    # n_next = 1

    type_num_normalization_path =  base_path + 'data/type_%s_normalization/' % type_num
    sum_filename = 'type%s_%s' % (type_num, type_num)
    file_names_list = find_files(type_num_normalization_path)

    if sum_flag:
        print('Train test split of sum file...')
        file_names_list = [sum_filename]
    else:
        print('Train test split of single file...')
        file_names_list.remove(sum_filename)

    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        print('-------' + co_name + '--------')
        print('-------' + user_id + '--------')

        # 导入行业x的数据
        df = pd.read_csv(type_num_normalization_path + file_name + '.csv')
        df.drop(['data_date'], axis=1, inplace=True)

        data = np.array(df)

        load = df['load'].values
        load = load.reshape(len(load), 1)

        train_data = data[:96 * train_range, :]  # (31+28+31) ｜ 365 ｜ （365+31+28) * 0.8 = 339.2 = 340
        train_load = load[:96 * train_range, :]
        test_data = data[96 * train_range: 96 * (train_range + test_range), :]
        test_load = load[96 * train_range: 96 * (train_range + test_range), :]

        train_x, _ = create_dataset(train_data, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
        _, train_y = create_dataset(train_load, n_predictions, n_next)
        test_x, _ = create_dataset(test_data, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
        _, test_y = create_dataset(test_load, n_predictions, n_next)

        # # 只有负荷
        # train_x, _ = create_dataset(train_load, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
        # _, train_y = create_dataset(train_load, n_predictions, n_next)
        # test_x, _ = create_dataset(test_load, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
        # _, test_y = create_dataset(test_load, n_predictions, n_next)

        co_name_user_id_path = base_path + 'data/train_test/type%s/%s_%s/' % (type_num, co_name, user_id)
        if not os.path.exists(co_name_user_id_path):
            os.makedirs(co_name_user_id_path)

        np.save(co_name_user_id_path + 'train_x_days_%s.npy' % train_range, train_x)
        np.save(co_name_user_id_path + 'train_y_days_%s.npy' % train_range, train_y)
        np.save(co_name_user_id_path + 'test_x_days_%s.npy' % train_range, test_x)
        np.save(co_name_user_id_path + 'test_y_days_%s.npy' % train_range, test_y)

    # 测试划分正确性
    # np.random.seed(0)
    # values = np.random.rand(20, 3)
    # label = values[:, -1].reshape(20, 1)
    # # data = pd.DataFrame(values, columns=['a', 'b', 'c'])
    #
    # train_x, _ = create_dataset(values, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
    # _, train_y = create_dataset(label, n_predictions, n_next)
    #
    # print(train_x)
    # print(train_y)