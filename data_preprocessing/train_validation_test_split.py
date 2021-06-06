import pandas as pd
import numpy as np
import os
from .utils import find_files
from .utils import create_dataset


def train_validation_test_split(base_path, args, sum_flag=False):
    type_num_normalization_path = base_path + 'data/type_%s/day_%s/%s_normalization/' % (args.type_num, args.day_range, args.norm)
    sum_filename = 'type%s_%s' % (args.type_num, args.type_num)
    file_names_list = find_files(type_num_normalization_path)
    if sum_flag:
        print('Train validation test split of sum file...')
        file_names_list = [sum_filename]
    else:
        print('Train validation test split of single file...')
        try:
            file_names_list.remove(sum_filename)
        except:
            pass

    n_predictions = args.seq_len
    n_next = args.out_features

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

        if args.train_range == 426:
            train_range = 426
            validation_range = 122
            test_range = 61
        elif args.train_range == 512:
            train_range = 512
            validation_range = 122
            test_range = 73
        else:
            train_range = int((365 + 366) * 0.6)
            validation_range = int((365 + 366) * 0.2)
            test_range = int((365 + 366) * 0.2)

        print('train_range:', train_range)
        print('validation_range:', validation_range)
        print('test_range:', test_range)

        train_data = data[:args.day_range * train_range, :]
        train_load = load[:args.day_range * train_range, :]
        validation_data = data[args.day_range * train_range: args.day_range * (train_range + validation_range), :]
        validation_load = load[args.day_range * train_range: args.day_range * (train_range + validation_range), :]
        test_data = data[args.day_range * (train_range + validation_range): args.day_range * (train_range + validation_range + test_range), :]
        test_load = load[args.day_range * (train_range + validation_range): args.day_range * (train_range + validation_range + test_range), :]

        # train_x, _ = create_dataset(train_data, n_predictions, n_next)  # n_predictions个点预测n_next个点
        # _, train_y = create_dataset(train_load, n_predictions, n_next)
        # validation_x, _ = create_dataset(validation_data, n_predictions, n_next)
        # _, validation_y = create_dataset(validation_load, n_predictions, n_next)
        # test_x, _ = create_dataset(test_data, n_predictions, n_next)
        # _, test_y = create_dataset(test_load, n_predictions, n_next)

        train_x, _ = create_dataset(train_data, n_predictions, n_next)  # n_predictions个点预测n_next个点
        _, train_y = create_dataset(train_data, n_predictions, n_next)
        validation_x, _ = create_dataset(validation_data, n_predictions, n_next)
        _, validation_y = create_dataset(validation_data, n_predictions, n_next)
        test_x, _ = create_dataset(test_data, n_predictions, n_next)
        _, test_y = create_dataset(test_data, n_predictions, n_next)

        normalization_tvt_path = base_path + 'data/type_%s/day_%s/%s_normalization_tvt/%s_%s/' \
                                 % (args.type_num, args.day_range, args.norm, co_name, user_id)

        if not os.path.exists(normalization_tvt_path):
            os.makedirs(normalization_tvt_path)

        print('Saving %s - %s npy file...' % (co_name, user_id))
        np.save(normalization_tvt_path + 'train_x_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), train_x)
        np.save(normalization_tvt_path + 'train_y_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), train_y)
        np.save(normalization_tvt_path + 'validation_x_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), validation_x)
        np.save(normalization_tvt_path + 'validation_y_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), validation_y)
        np.save(normalization_tvt_path + 'test_x_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), test_x)
        np.save(normalization_tvt_path + 'test_y_in_%s_out_%s_range_%s.npy'
                % (n_predictions, n_next, train_range), test_y)
