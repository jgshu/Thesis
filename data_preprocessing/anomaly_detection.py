import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from scipy import stats
from .utils import find_files


def anomaly_replace(data, df_len, err_idx_list, detection_num):
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
            while (right in err_idx_list) & (right < df_len):
                right = right + 1
                cnt = cnt + 1
            if right == len(data):
                print('Fail.')
                break
            for j in range(1, cnt + 1):
                # 存在替换后仍为异常值的问题，如type12的南京聚隆科技股份有限公司
                if detection_num < 15:
                    data.loc[num + j - 1] = (j - 1) * data.loc[right] / right
                else:
                    data.loc[num + j - 1] = data.loc[right]
                # data.loc[num + j - 1] =  data.loc[num + j - 1 + 96 * 7]
            i = i + cnt

        elif num == df_len - 1:
            left = num - 1
            data.loc[num] = data.loc[left]

    return data


def three_sigma_alg(anomaly_detection_path, df, user_id, co_name):
    data = df['load']
    # 创建数据
    u = data.mean()  # 计算均值
    std = data.std()  # 计算标准差
    error = data[np.abs(data - u) > 3 * std]  # 超过3倍差的数据（即异常值）筛选出来
    err_idx_list = list(error.index)
    stats.kstest(data, 'norm', (u, std))
    # print('------')
    # print('Mean：%.3f，Std：%.3f' % (u, std))
    # print('------')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 50,
            }
    detection_num = 1
    while len(err_idx_list) > 0:
        # 正态性检验
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        data.plot(kind='kde', grid=True, style='-k', title='Density curve')
        plt.axvline(3 * std, color='r', linestyle="--", alpha=0.8)  # 3倍的标准差
        plt.axvline(-3 * std, color='r', linestyle="--", alpha=0.8)

        # 绘制数据密度曲线
        data_c = data[np.abs(data - u) < 3 * std]  # 正常数据
        print('* Before Anomaly Detection: %i' % len(error))
        ax2 = fig.add_subplot(2, 1, 2)

        plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
        plt.scatter(error.index, error, color='r', marker='.', alpha=0.7)
        plt.xlim([-10, 73000])
        plt.grid()

        # 保存图片
        plt.savefig(anomaly_detection_path + '%s_%s_%d.png' % (co_name, user_id, detection_num))
        plt.close(fig)

        # 前后取均值替换异常数据

        data = anomaly_replace(data, len(df), err_idx_list, detection_num)

        # 再次检测
        u = data.mean()
        std = data.std()
        error = data[np.abs(data - u) > 3 * std]
        err_idx_list = list(error.index)
        detection_num = detection_num + 1
        print('* After Anomaly Detection: %i' % len(error))

    return data


def anomaly_detection(base_path, type_num, anomaly_detection_path, sum_flag=False, need_ad=True):
    type_num_original_path = base_path + 'data/type_%s/original/' % type_num
    type_num_after_anomaly_detection_path = base_path + 'data/type_%s/after_anomaly_detection/' % type_num

    sum_filename = 'type%s_%s' % (type_num, type_num)
    file_names_list = find_files(type_num_original_path)

    if sum_flag:
        print('Anomaly detection of sum file...')
        file_names_list = [sum_filename]
    else:
        print('Anomaly detection of single file...')
        try:
            file_names_list.remove(sum_filename)
        except:
            pass

    for file_name in file_names_list:
        co_name, user_id = file_name.split('_')
        print('-------' + co_name + '--------')
        print('-------' + user_id + '--------')

        # 导入行业x的数据
        df = pd.read_csv(type_num_original_path + file_name + '.csv')

        data = df.iloc[:, 1:].values
        print('Any nan?', np.any(np.isnan(data)))
        df = df.fillna(method='ffill')
        data = df.iloc[:, 1:].values
        print('Any nan?', np.any(np.isnan(data)))

        if need_ad:
            # 异常值分析 - 3σ原则
            data = three_sigma_alg(anomaly_detection_path, df, user_id, co_name)
        else:
            data = df['load']

        # 保存csv文件
        if not os.path.exists(type_num_after_anomaly_detection_path):
            os.makedirs(type_num_after_anomaly_detection_path)
        df['load'] = data
        df.to_csv(type_num_after_anomaly_detection_path + '%s_%s.csv' % (co_name, user_id), index=False)