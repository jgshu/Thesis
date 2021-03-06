import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_preprocessing.utils import find_files
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


def get_dict(type_num_path, day_range, need_slice=False):
    file_names_list = find_files(type_num_path)

    user_id_co_name_dict= {}
    user_id_df_dict = {}
    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        # 导入行业x的数据
        df = pd.read_csv(type_num_path + file_name + '.csv')
        df.drop(['data_date'], axis=1, inplace=True)
        if need_slice:
            slice_range = int(96 / day_range)
            df = df[::slice_range]
        user_id_co_name_dict[user_id] = co_name
        user_id_df_dict[user_id] = df

    return user_id_df_dict, user_id_co_name_dict


def plot_by_per_day(daily_load_path, user_id_df_dict, user_id_co_name_dict, font, day_range, start=1, end=40, per_day=7):
    user_id_list = list(user_id_df_dict.keys())
    left = 0

    if per_day == 7:
        right = int((end - start) / 7) + 1
    elif per_day == 1:
        right = end - start + 1

    for k in range(left, right):
        if per_day == 7:
            print('week:', k - left + 1)
        elif per_day == 1:
            print('day:', k - left + 1)
        i = 1
        fig = plt.figure()
        fig.set_size_inches(60, 80, forward=True)
        for user_id in user_id_df_dict.keys():
            co_name = user_id_co_name_dict[user_id]
            user_df = user_id_df_dict[user_id].copy()
            data = user_df['load']

            ax = plt.subplot(len(user_id_list), 1, i)
            y = data.values
            y = y[start - 1 + day_range * per_day * k: start - 1 + day_range * per_day * (k + 1)]
            # y = y[day_range * per_day * k: day_range * per_day * (k + 1)]
            x = range(0, len(y))
            ax.plot(x, y)
            plt.xticks(font=font)
            plt.yticks(font=font)
            ax.set_ylabel('%s' % co_name, font=font)
            ax.set_xticks(np.arange(0, len(y), day_range))
            ax.set_xticklabels(np.arange(0, int(len(y) / day_range)))
            i = i + 1

        if not os.path.exists(daily_load_path):
            os.makedirs(daily_load_path)

        if per_day == 7:
            plt.savefig(daily_load_path + 'from_%s_to_%s_week_%02d.jpg' % (start, end, k - left + 1))
        elif per_day == 1:
            plt.savefig(daily_load_path + 'from_%s_to_%s_day_%02d.jpg' % (start, end, k - left + 1))
        plt.close(fig)


def daily_load_plotting(base_path, args, start=1, end=40, per_day=7, need_filter=False):
    font = {
        'family': 'SimHei',
        'weight': 'normal',
        # 'size': 10,
       }

    if need_filter:
        # 导入数据
        type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % (args.type_num)
        user_id_df_dict, user_id_co_name_dict = get_dict(type_num_after_anomaly_detection_path, args.day_range)

        daily_load_path = base_path + 'output/type_%s/day_%s_range_%s_%s/img/daily_load_filter/' \
                 % (args.type_num, args.day_range, args.train_range, args.norm)
        plot_by_per_day(daily_load_path, user_id_df_dict, user_id_co_name_dict, font, args.day_range,
                        start=start, end=end, per_day=per_day)
    else:
        # 导入数据
        print('')
        type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % args.type_num
        user_id_df_dict, user_id_co_name_dict = get_dict(type_num_after_anomaly_detection_path, args.day_range,
                                                         need_slice=True)

        daily_load_path = base_path + 'output/type_%s/day_%s_range_%s_%s/img/daily_load/after_anomaly_detection/' \
                 % (args.type_num, args.day_range, args.train_range, args.norm)
        plot_by_per_day(daily_load_path, user_id_df_dict, user_id_co_name_dict, font, args.day_range,
                           start=start, end=end, per_day=per_day)
        # 导入数据
        type_num_normalization_path = base_path + 'data/type_%s/day_%s/%s_normalization/' \
                                      % (args.type_num, args.day_range, args.norm)
        user_id_df_dict, user_id_co_name_dict = get_dict(type_num_normalization_path, args.day_range)

        daily_load_path = base_path + 'output/type_%s/day_%s_range_%s_%s/img/daily_load/normalization/' \
                 % (args.type_num, args.day_range, args.train_range, args.norm)
        plot_by_per_day(daily_load_path, user_id_df_dict, user_id_co_name_dict, font, args.day_range,
                           start=start, end=end, per_day=per_day)
