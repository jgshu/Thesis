import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def test_plotting(base_path, args, file_name, dt_string, test_y, test_y_pred, specific_user_id):
    font = {
        'family': 'SimHei',
        'weight': 'normal',
        # 'size': 20,
    }
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]

    n_fig = 24
    fig = plt.figure()
    fig.set_size_inches(30, 60, forward=True)
    for i in range(n_fig):
        j = args.day_range * i
        ax = plt.subplot(n_fig, 1, i + 1)
        y1 = test_y[j: j + args.day_range: args.day_range, :]
        y1 = y1.reshape(y1.shape[0] * y1.shape[1])
        y2 = test_y_pred[j: j + args.day_range: args.day_range, :]
        y2 = y2.reshape(y2.shape[0] * y2.shape[1])
        x = range(0, y1.shape[0])
        ax.plot(x, y1, color_list[0])
        ax.plot(x, y2, color_list[-1])
        plt.xticks(font=font)
        plt.yticks(font=font)
        ax.set_ylabel('values_day%s' % (i + 1), font=font)
        ax.set_xticks(np.arange(0, y1.shape[0], args.day_range))
        ax.set_xticklabels(np.arange(0, int(y1.shape[0] / args.day_range)))

    test_path = base_path + 'output/img/type_%s/test/day_%s/%s_normalization/%s/' % (args.type_num, args.day_range, args.norm, dt_string)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    plt.savefig(test_path + 'stacked_%s_%s.jpg' % (specific_user_id, file_name))
    # plt.show()

    i = 0
    y1 = test_y[i::args.day_range, :]
    y1 = y1.reshape(y1.shape[0] * y1.shape[1])

    y2 = test_y_pred[i::args.day_range, :]
    y2 = y2.reshape(y2.shape[0] * y2.shape[1])

    fig, ax = plt.subplots(figsize=(50, 30))
    x = range(0, y1.shape[0])
    ax.plot(x, y1, color_list[0])
    ax.plot(x, y2, color_list[-1])
    plt.xticks(font=font)
    plt.yticks(font=font)
    ax.set_ylabel('values', font=font)
    ax.set_xticks(np.arange(0, y1.shape[0], args.day_range))
    ax.set_xticklabels(np.arange(0, int(y1.shape[0] / args.day_range)))
    plt.savefig(test_path + 'single_%s_%s.jpg' % (specific_user_id, file_name))
    # plt.show()
