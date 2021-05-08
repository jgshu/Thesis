import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit


def create_dataset(data, n_predictions, n_next):
    """
    对数据进行处理
    :param data:
    :param n_predictions:
    :param n_next:
    :return:
    """
    dim = data.shape[1]
    train_x, train_y  = [], []
    for i in range(data.shape[0] - n_predictions - n_next + 1):
        a = data[i: (i + n_predictions), :]
        train_x.append(a)
        tempb = data[(i + n_predictions): (i + n_predictions + n_next), :]
        b = []
        for j in range(len(tempb)):
            for k in range(dim):
                b.append(tempb[j, k])
        train_y.append(b)
    train_x = np.array(train_x, dtype='float64')
    train_y = np.array(train_y, dtype='float64')
    return train_x, train_y

# 可调参数
type_num = 3
date_range = 90
n_predictions = 96 * 7
n_next = 96
# n_predictions = 96
# n_next = 1

# 导入数据
csv_path = './data/type_%s_csv_normalization/' % type_num
file_list = os.listdir(csv_path)
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    # print(coName)
    user_id_co_name_dict[userID] = coName
    df = pd.read_csv(csv_path + file)
    df.drop(['data_date'], axis=1, inplace=True)
    user_id_df_dict[userID] = df

user_id_list = list(user_id_df_dict.keys())

# 训练集和测试集
for userID in user_id_df_dict.keys():
    coName = user_id_co_name_dict[userID]
    user_df = user_id_df_dict[userID].copy()
    data = np.array(user_df)
    print(coName)
    load = user_df['load'].values
    load = load.reshape(len(load), 1)

    if date_range == 90:
        train_data = data[:10752, :]  # 96 * (31 + 28 +31) =
        train_load = load[:10752, :]
        test_data = data[10752:13440, :]
        test_load = load[10752:13440, :]
    elif date_range == 365:
        train_data = data[:35040, :]  # 96 * 365
        train_load = load[:35040, :]
        test_data = data[35040:38016, :]
        test_load = load[35040:38016, :]
    else:
        train_data = data[:32640, :]  # 96 * ((365+31+28) * 0.8 = 339.2 = 340) = 32640
        train_load = load[:32640, :]
        test_data = data[32640:40704, :]
        test_load = load[32640:40704, :]

    train_x, _ = create_dataset(train_data, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
    _, train_y = create_dataset(train_load, n_predictions, n_next)
    test_x, _ = create_dataset(test_data, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
    _, test_y = create_dataset(test_load, n_predictions, n_next)

    # # 只有负荷
    # train_x, _ = create_dataset(train_load, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
    # _, train_y = create_dataset(train_load, n_predictions, n_next)
    # test_x, _ = create_dataset(test_load, n_predictions, n_next)  # 前一周96 * 7个点预测当天96个点
    # _, test_y = create_dataset(test_load, n_predictions, n_next)

    path = './output/train_test/type%s/%s_%s/' % (type_num, coName, userID)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(path + 'train_x_days_%s.npy' % date_range, train_x)
    np.save(path + 'train_y_days_%s.npy' % date_range, train_y)
    np.save(path + 'test_x_days_%s.npy' % date_range, test_x)
    np.save(path + 'test_y_days_%s.npy' % date_range, test_y)

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