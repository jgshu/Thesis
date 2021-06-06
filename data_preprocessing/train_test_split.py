import pandas as pd
import numpy as np
import os
from .utils import find_files
from .utils import create_dataset


def train_test_split(base_path, type_num, n_predictions, n_next, train_range, test_range, day_range=96, norm='minmax', sum_flag=False):
    type_num_normalization_path = base_path + 'data/type_%s/day_%s/%s_normalization/' % (type_num, day_range, norm)
    sum_filename = 'type%s_%s' % (type_num, type_num)
    file_names_list = find_files(type_num_normalization_path)

    if sum_flag:
        print('Train test split of sum file...')
        file_names_list = [sum_filename]
    else:
        print('Train test split of single file...')
        try:
            file_names_list.remove(sum_filename)
        except:
            pass

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

        train_data = data[:day_range * train_range, :]  # (31+28+31) ｜ 365 ｜ （365+31+28) * 0.8 = 339.2 = 340
        train_load = load[:day_range * train_range, :]
        test_data = data[day_range * train_range: day_range * (train_range + test_range), :]
        test_load = load[day_range * train_range: day_range * (train_range + test_range), :]

        train_x, _ = create_dataset(train_data, n_predictions, n_next)  # n_predictions个点预测n_next个点
        _, train_y = create_dataset(train_load, n_predictions, n_next)
        test_x, _ = create_dataset(test_data, n_predictions, n_next)
        _, test_y = create_dataset(test_load, n_predictions, n_next)

        normalization_tt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tt/%s_%s/' % (
        type_num, day_range, norm, co_name, user_id)
        if not os.path.exists(normalization_tt_path):
            os.makedirs(normalization_tt_path)

        print('Saving %s - %s npy file...' % (co_name, user_id))
        np.save(normalization_tt_path + 'train_x_in_%s_out_%s_range_%s.npy' % (n_predictions, n_next, train_range), train_x)
        np.save(normalization_tt_path + 'train_y_in_%s_out_%s_range_%s.npy' % (n_predictions, n_next, train_range), train_y)
        np.save(normalization_tt_path + 'test_x_in_%s_out_%s_range_%s.npy' % (n_predictions, n_next, train_range), test_x)
        np.save(normalization_tt_path + 'test_y_in_%s_out_%s_range_%s.npy' % (n_predictions, n_next, train_range), test_y)
