import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats


# csv_original_path = './data/type_10_csv_original/'
# file_list = os.listdir(csv_original_path)
# user_id_df_dict = {}
# user_id_co_name_dict = {}
# for i in range(len(file_list)):
#     file = file_list[i]
#     coName, userID = file[:-4].split('_')
#     print(coName)
#     user_id_co_name_dict[userID] = coName
#     df = pd.read_csv(csv_original_path + file)
#     user_id_df_dict[userID] = df
type_num = 3
csv_original_path = './data/type_%s_csv_original/' % type_num
file_list = os.listdir(csv_original_path)
user_id_df_dict = {}
user_id_co_name_dict = {}
for i in range(len(file_list)):
    file = file_list[i]
    coName, userID = file[:-4].split('_')
    print(coName)
    user_id_co_name_dict[userID] = coName
    df = pd.read_csv(csv_original_path + file)
    data = df.iloc[:, 1:].values
    print(np.any(np.isnan(data)))
    df = df.fillna(method='ffill')
    data = df.iloc[:, 1:].values
    print(np.any(np.isnan(data)))
    user_id_df_dict[userID] = df

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 50,
        }
# 异常值分析
# 3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值 → p(|x - μ| > 3σ) ≤ 0.003
for userID in user_id_df_dict.keys():
    coName = user_id_co_name_dict[userID]
    user_df = user_id_df_dict[userID].copy()
    data = user_df['load']
    # 创建数据
    u = data.mean()  # 计算均值
    std = data.std()  # 计算标准差
    error = data[np.abs(data - u) > 3 * std]  # 超过3倍差的数据（即异常值）筛选出来
    err_idx_list = list(error.index)
    stats.kstest(data, 'norm', (
    u, std))  # 正态分布的方式，得到 KstestResult(statistic=0.012627414595288711, pvalue=0.082417721086262413)，P值>0.5
    print('------')
    print('User ID：%s，Co Name：%s' % (userID, coName))
    print('Mean：%.3f，Std：%.3f' % (u, std))
    print('------')

    detection_num = 1
    while len(err_idx_list) > 0:
        # 正态性检验
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        data.plot(kind='kde', grid=True, style='-k', title='Density curve')
        plt.axvline(3 * std, color='r', linestyle="--", alpha=0.8)  # 3倍的标准差
        plt.axvline(-3 * std, color='r', linestyle="--", alpha=0.8)

        # 绘制数据密度曲线
        data_c = data[np.abs(data - u) < 3 * std] # 正常数据
        print('* Before Anomaly Detection: %i' % len(error))
        ax2 = fig.add_subplot(2, 1, 2)

        plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
        plt.scatter(error.index, error, color='r', marker='.', alpha=0.7)
        plt.xlim([-10, 73000])
        plt.grid()
        # 保存图片
        plt.savefig('./output/img/anomaly detection/%s_%s_ad_%d.png' % (coName, userID, detection_num))
        # plt.show()
        # 前后取均值替换异常数据

        i = 0
        while i < len(err_idx_list):
            num = err_idx_list[i]
            right = num + 1
            if (num > 0) & (num < len(data) - 1):
                left = num - 1
                cnt = 1
                while (right in err_idx_list) & (right < len(data)):
                    right = right + 1
                    cnt = cnt + 1
                if right == len(data):
                    print('Fail.')
                    break
                for j in range(1, cnt + 1):
                    data.loc[num + j - 1] = data.loc[left] + j * (data.loc[right] - data.loc[left]) / (right - left)
                i = i + cnt
            elif num == 0:
                right = num + 1
                cnt = 1
                while (right in err_idx_list) & (right < len(user_df)):
                    cnt = cnt + 1
                if right == len(data):
                    print('Fail.')
                    break
                for j in range(1, cnt + 1):
                    data.loc[num + j - 1] = (j - 1) * data.loc[right] / right
                i = i + cnt
            elif num == len(user_df) - 1:
                left = num - 1
                data.loc[num] = data.loc[left]
        u = data.mean()  # 再次计算均值
        std = data.std()  # 再次计算标准差
        error = data[np.abs(data - u) > 3 * std]
        err_idx_list = list(error.index)
        detection_num = detection_num + 1
    print('* After Anomaly Detection: %i' % len(error))

    # 保存csv文件
    path = './data/type_%s_csv_after_anomaly_detection/' % type_num
    if not os.path.exists(path):
        os.makedirs(path)
    user_df['load'] = data
    user_df.to_csv(path + '%s_%s.csv' % (coName, userID), index=False)