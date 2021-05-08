import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

type_num = 10

# 导入数据
csv_path = './data/type_%s_csv_normalization/' % type_num
file_list = os.listdir(csv_path)
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    print(coName)
    user_id_co_name_dict[userID] = coName
    df = pd.read_csv(csv_path + file)
    user_id_df_dict[userID] = df

user_id_list = list(user_id_df_dict.keys())

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 50,
       }

# end = 40
end = 2
for day in range(1, end):
    i = 1
    fig = plt.figure()
    fig.set_size_inches(50, 50, forward=True)
    for userID in user_id_df_dict.keys():
        coName = user_id_co_name_dict[userID]
        user_df = user_id_df_dict[userID].copy()
        data = user_df['load']

        ax = plt.subplot(len(user_id_list), 1, i)
        y = data.values
        # y = y[96 * (day - 1): 96 * day]
        y = y[96 * 7 * (day - 1): 96 * 7 * day]
        x = range(0, len(y))
        ax.plot(x, y)
        plt.xticks(font=font)
        plt.yticks(font=font)
        ax.set_ylabel('%s' % userID, font=font)
        # ax.set_xticks(np.arange(0, len(y), 4))
        ax.set_xticks(np.arange(0, len(y), 96))
        ax.set_xticklabels(np.arange(0, 7))
        i = i + 1

    path = './output/img/day_load/type%s/' % type_num
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'day_%02d_end_%s.jpg' % (day, end))
    plt.show()
